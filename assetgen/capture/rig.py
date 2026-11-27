"""A glTF skin as a differentiable function of joint rotations: forward kinematics and linear blend skinning.

Each joint's pose is its rest rotation followed by a rotation of its own (axis-angle, in the joint's
frame). One joint, the hub (where the skeleton first branches: the hips of most rigs), also moves by
a translation in its parent's frame. Bone lengths never change.
"""
from __future__ import annotations

import numpy as np
import torch


def skew(w: torch.Tensor) -> torch.Tensor:
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*w.shape[:-1], 3, 3)


def axis_angle_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, (..., 3) -> (..., 3, 3), with a Taylor expansion near zero."""
    t2 = (w * w).sum(-1)[..., None, None]
    small = t2 < 1e-8
    t2_safe = torch.where(small, torch.ones_like(t2), t2)
    t = torch.sqrt(t2_safe)
    a = torch.where(small, 1 - t2 / 6, torch.sin(t) / t)
    b = torch.where(small, 0.5 - t2 / 24, (1 - torch.cos(t)) / t2_safe)
    W = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(W)
    return eye + a * W + b * (W @ W)


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 4) unit quaternions in glTF order (x, y, z, w), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    q = np.stack([
        np.sqrt(np.maximum(0, 1 + m00 - m11 - m22)) / 2,
        np.sqrt(np.maximum(0, 1 - m00 + m11 - m22)) / 2,
        np.sqrt(np.maximum(0, 1 - m00 - m11 + m22)) / 2,
        np.sqrt(np.maximum(0, 1 + m00 + m11 + m22)) / 2,
    ], -1)
    q[..., 0] = np.copysign(q[..., 0], R[..., 2, 1] - R[..., 1, 2])
    q[..., 1] = np.copysign(q[..., 1], R[..., 0, 2] - R[..., 2, 0])
    q[..., 2] = np.copysign(q[..., 2], R[..., 1, 0] - R[..., 0, 1])
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*np.shape(q)[:-1], 3, 3)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    q = matrix_to_quaternion(R)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, q[..., 3:])
    return np.where(s > 1e-12, q[..., :3] / np.maximum(s, 1e-12) * angle, 2 * q[..., :3])


class RigModel:
    """The skin of a skinned glTF (an :class:`assetgen.gltf_anim.AnimatedGLB`) as a torch function."""

    def __init__(self, g, device: str | torch.device = "cpu", dtype=torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        self.nodes = list(g.skin.joints)
        self.names = [g.doc.nodes[n].name or f"node_{n}" for n in self.nodes]
        index = {n: i for i, n in enumerate(self.nodes)}
        rest_world = g.world_matrices(None)
        trs = g._local_trs(None, 0.0)

        def depth(n):
            d = 0
            while n in g.parent and g.parent[n] in index:
                n, d = g.parent[n], d + 1
            return d

        self.order = sorted(range(len(self.nodes)), key=lambda i: depth(self.nodes[i]))
        self.parent = [index.get(g.parent.get(n), -1) for n in self.nodes]
        self.hub = self._hub()
        base = []
        for n in self.nodes:
            p = g.parent.get(n)
            base.append(rest_world[p] if p is not None and p not in index else np.eye(4))
        from ..gltf_anim import _quat_to_matrix

        t = lambda a: torch.as_tensor(np.array(a), dtype=dtype, device=self.device)  # noqa: E731
        self.base = t(np.stack(base))
        self.rest_translation = t(np.stack([trs[n][0] for n in self.nodes]))
        self.rest_rotation = t(np.stack([_quat_to_matrix(trs[n][1]) for n in self.nodes]))
        self.rest_scale = t(np.stack([trs[n][2] for n in self.nodes]))
        self.ibm = t(g.ibm)
        self.vertices = np.asarray(g.positions, dtype=np.float64)
        self.faces = np.asarray(g.faces)
        self.vertex_joints = np.asarray(g.joints)
        self.vertex_weights = np.asarray(g.weights, dtype=np.float64)
        rest = g.posed(None)
        self.size = float(np.ptp(rest, axis=0).max()) or 1.0

    def _hub(self) -> int:
        """The joint that carries the body's translation: from the root with the most descendants, down
        its chain to the first joint that branches (a hip under a root and a ground bone, say); a chain
        that never branches keeps its root."""
        children = [[c for c, p in enumerate(self.parent) if p == j] for j in range(len(self.parent))]

        def count(j):
            return 1 + sum(count(c) for c in children[j])

        j = max((i for i, p in enumerate(self.parent) if p < 0), key=count)
        while len(children[j]) == 1 and children[children[j][0]]:  # never down to a leaf: a plain chain keeps its root
            j = children[j][0]
        return j

    @property
    def n_joints(self) -> int:
        return len(self.nodes)

    def zeros(self, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A pose sequence at rest: rotations (T, J, 3) and hub translations (T, 3)."""
        return (torch.zeros(frames, self.n_joints, 3, dtype=self.dtype, device=self.device),
                torch.zeros(frames, 3, dtype=self.dtype, device=self.device))

    def world(self, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Joint world matrices (T, J, 4, 4) for rotations (T, J, 3) and hub translations (T, 3), the
        latter in the hub's parent frame."""
        T = rotation.shape[0]
        R = self.rest_rotation[None] @ axis_angle_to_matrix(rotation)            # (T, J, 3, 3)
        R = R * self.rest_scale[None, :, None, :]
        trans = self.rest_translation[None].expand(T, -1, -1)
        offset = torch.zeros_like(trans)
        offset[:, self.hub] = translation
        trans = trans + offset
        local = torch.zeros(T, self.n_joints, 4, 4, dtype=self.dtype, device=self.device)
        local[..., :3, :3] = R
        local[..., :3, 3] = trans
        local[..., 3, 3] = 1
        world = [None] * self.n_joints
        for j in self.order:
            p = self.parent[j]
            parent = self.base[j][None] if p < 0 else world[p]
            world[j] = parent @ local[:, j]
        return torch.stack(world, 1)

    def skinning(self, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """Skinning matrices (T, J, 4, 4): joint world matrices times inverse bind matrices."""
        return self.world(rotation, translation) @ self.ibm[None]

    def point_basis(self, faces: np.ndarray, bary: np.ndarray) -> torch.Tensor:
        """Per-joint homogeneous rest positions (N, J, 4) of surface points given as (face, barycentric):
        a point's skinned position is sum_j M_j u_j, exactly as linear blend skinning moves the triangle."""
        verts = self.faces[np.asarray(faces)]                                      # (N, 3)
        return self._basis(verts, np.asarray(bary, dtype=np.float64))

    def vertex_basis(self, vertices: np.ndarray | None = None) -> torch.Tensor:
        idx = np.arange(len(self.vertices)) if vertices is None else np.asarray(vertices)
        return self._basis(idx[:, None], np.ones((len(idx), 1)))

    def _basis(self, verts: np.ndarray, bary: np.ndarray) -> torch.Tensor:
        N, K = verts.shape
        u = np.zeros((N, self.n_joints, 4))
        rows = np.repeat(np.arange(N), K * self.vertex_joints.shape[1])
        hom = np.concatenate([self.vertices, np.ones((len(self.vertices), 1))], 1)
        joints = self.vertex_joints[verts].reshape(N, -1)                          # (N, K*4)
        coef = (bary[:, :, None] * self.vertex_weights[verts]).reshape(N, -1)      # (N, K*4)
        pos = np.repeat(hom[verts], self.vertex_joints.shape[1], axis=1)           # (N, K*4, 4)
        np.add.at(u, (rows, joints.ravel()), (coef[..., None] * pos).reshape(-1, 4))
        return torch.as_tensor(u, dtype=self.dtype, device=self.device)

    def pose_points(self, basis: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor,
                    chunk: int = 20000) -> torch.Tensor:
        """World positions (T, N, 3) of the points behind ``basis`` (see :meth:`point_basis`)."""
        M = self.skinning(rotation, translation)[..., :3, :]                        # (T, J, 3, 4)
        return torch.cat([torch.einsum("tjab,njb->tna", M, basis[i:i + chunk])
                          for i in range(0, len(basis), chunk)], 1)

    def local_rotations(self, rotation: torch.Tensor) -> np.ndarray:
        """Each joint's local rotation (T, J, 3, 3): rest rotation, then the fitted one."""
        return (self.rest_rotation[None] @ axis_angle_to_matrix(rotation)).detach().cpu().double().numpy()


def from_animation(rig: RigModel, g, anim: int, times) -> tuple[torch.Tensor, torch.Tensor]:
    """The (rotation, translation) parameters that reproduce animation ``anim`` of ``g`` on ``rig``
    (``g`` and ``rig`` share a skeleton), at ``times`` seconds. Translations of joints other than the
    hub, and scales, are not representable and are left at rest."""
    rot, tr = [], []
    rest_R = rig.rest_rotation.double().cpu().numpy()
    rest_T = rig.rest_translation.double().cpu().numpy()
    for t in times:
        trs = g._local_trs(anim, float(t))
        R = quaternion_to_matrix(np.stack([trs[n][1] for n in rig.nodes]))
        rot.append(matrix_to_axis_angle(np.transpose(rest_R, (0, 2, 1)) @ R))
        tr.append(trs[rig.nodes[rig.hub]][0] - rest_T[rig.hub])
    return (torch.as_tensor(np.stack(rot), dtype=rig.dtype, device=rig.device),
            torch.as_tensor(np.stack(tr), dtype=rig.dtype, device=rig.device))
