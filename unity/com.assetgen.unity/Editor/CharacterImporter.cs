using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>
    /// Imports a <c>.assetgen</c> file: a character as a <see cref="CharacterDefinition"/>, a prop or
    /// diorama as a <see cref="PropDefinition"/>.
    /// </summary>
    [ScriptedImporter(2, "assetgen")]
    sealed class CharacterImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            ScriptableObject definition;
            string identifier;
            Texture2D icon;
            try
            {
                var json = File.ReadAllText(ctx.assetPath);
                if (FormatOf(json) == PropDefinition.Format)
                {
                    definition = PropDefinition.FromJson(json);
                    identifier = "prop";
                    icon = AssetPreview.GetMiniTypeThumbnail(typeof(MeshFilter));
                }
                else
                {
                    definition = CharacterDefinition.FromJson(json);
                    identifier = "character";
                    icon = AssetPreview.GetMiniTypeThumbnail(typeof(Avatar));
                }
            }
            catch (Exception e) when (e is FormatException || e is ArgumentException)
            {
                ctx.LogImportError($"{ctx.assetPath}: {e.Message}");
                return;
            }
            definition.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            ctx.AddObjectToAsset(identifier, definition, icon);
            ctx.SetMainObject(definition);
        }

        /// <summary>The file's <c>format</c>. Throws <see cref="ArgumentException"/> when it is not JSON.</summary>
        internal static string FormatOf(string json) => JsonUtility.FromJson<Header>(json)?.format;

#pragma warning disable 0649
        [Serializable] class Header { public string format; }
#pragma warning restore 0649
    }
}
