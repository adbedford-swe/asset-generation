using System;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Assetgen.Editor
{
    /// <summary>Imports a <c>.assetgen</c> character file as a <see cref="CharacterDefinition"/>.</summary>
    [ScriptedImporter(1, "assetgen")]
    sealed class CharacterImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            CharacterDefinition definition;
            try
            {
                definition = CharacterDefinition.FromJson(File.ReadAllText(ctx.assetPath));
            }
            catch (Exception e) when (e is FormatException || e is ArgumentException)
            {
                ctx.LogImportError($"{ctx.assetPath}: {e.Message}");
                return;
            }
            definition.name = Path.GetFileNameWithoutExtension(ctx.assetPath);
            ctx.AddObjectToAsset("character", definition, AssetPreview.GetMiniTypeThumbnail(typeof(Avatar)));
            ctx.SetMainObject(definition);
        }
    }
}
