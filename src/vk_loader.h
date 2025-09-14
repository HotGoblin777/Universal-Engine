#pragma once
#include <vk_types.h>
#include <unordered_map>
#include <filesystem>

class VulkanEngine;

struct GLTFMaterial
{
	MaterialInstance data;
};

struct GeoSurafce 
{
	uint32_t startIndex;
	uint32_t count;
	std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset 
{
	std::string name;

	std::vector<GeoSurafce> surfaces;  //массив, так как мы можем разделить одну модельку на несколько частей
	GPUMeshBuffer meshBuffers;
};

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);
