import argparse
from pathlib import Path
from tools.anatomy_segmentation_tool import available_organs


def args_tools():
    parser = argparse.ArgumentParser()
    # add_arguments
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--host", type=str, default="localhost")
    return parser.parse_args()


path_to_tool_root = Path(__file__).parent

JUDGING_TOOLS = ["report_judge_tool", "green_tool", "f1_text_classifier_tool"]
SERVERS = {
    "report_judge_tool": {
        "script": str(path_to_tool_root / "llm_judge_tool.py"),
        "port": 8010,
        "env": "RadAgentMel",
        "device": "1,2",  # 1
        "target_node": 0,
    },
    "green_tool": {
        "script": str(path_to_tool_root / "green_tool.py"),
        "port": 8011,
        "env": "RadAgentMel",
        "device": 3,
        "target_node": 0,
    },
    "f1_text_classifier_tool": {
        "script": str(path_to_tool_root / "text_classifier_tool.py"),
        "port": 8012,
        "env": "RadAgentMel",
        "device": 3,
        "target_node": 0,
    },
    "disease_classifier_tool": {
        "script": str(path_to_tool_root / "disease_classifier.py"),
        "port": 7010,
        "env": "CT-Chat2",
        "device": 2,
        "target_node": 0,
    },
    "windowing_tool": {
        "script": str(path_to_tool_root / "windowing_tool.py"),
        "port": 7013,
        "env": "torch2",
        "device": 0,
    },
    "biggest_slice_selection_tool": {
        "script": str(path_to_tool_root / "biggest_slice_selection_tool.py"),
        "port": 7014,
        "env": "torch2",
        "device": 0,
    },
    "get_several_slices_from_segmentation": {
        "script": str(path_to_tool_root / "equidistant_slice_selection_seg.py"),
        "port": 7018,
        "env": "torch2",
        "device": 0,
    },
    "extract_slices_from_ct": {
        "script": str(path_to_tool_root / "select_ct_slices.py"),
        "port": 7017,
        "env": "torch2",
        "device": 0,
    },
    "slice_vqa_tool": {
        "script": str(path_to_tool_root / "slice_vqa_tool.py"),
        "port": 7016,
        "env": "RadAgentMel",
        "device": "0,1",
        "target_node": 1,
    },
    "anatomy_segmentation_tool": {
        "script": str(path_to_tool_root / "anatomy_segmentation_tool.py"),
        "port": 7011,
        "env": "torch2",
        "device": 2,
        "target_node": 1,
    },
    "effusion_segmentation_tool": {
        "script": str(path_to_tool_root / "effusion_segmentation_tool.py"),
        "port": 7012,
        "env": "torch2",
        "device": 2,
        "target_node": 1,
    },
    "ct_vqa_tool": {
        "script": str(path_to_tool_root / "ct_chat.py"),
        "port": 6009,
        "env": "torch2",
        "device": "2,3",  # device 2,3
        "target_node": 1,
    },
    "report_generation_tool": {
        "script": str(path_to_tool_root / "report_generator.py"),
        "port": 7006,
        "env": "torch2",
        "device": 3,
        "target_node": 0,
    },
}


ALL_TOOL_CONFIGS = [
    {
        "name": "ct_vqa_tool",
        "description": "A tool that can answer any question about a CT scan.",
        "parameter": {
            "query": {"type": "string", "description": "Question about the CT image."},
            "image_path": {
                "type": "string",
                "description": "The path of the CT image.",
            },
        },
    },
    {
        "name": "report_generation_tool",
        "description": "Report generation model, it's only for whole image report generation, which may not be precise for specific region report generation. You can directly generate the report of the CT image using this tool when you finish checking all the specific regions of the scan according to the plan/checklist and want to have a initial but not very accurate draft to modify.",
        "parameter": {
            "query": {"type": "string", "description": "Question about the image."},
            "image_path": {
                "type": "string",
                "description": "The path of the CT image.",
            },
        },
    },
    {
        "name": "disease_classifier_tool",
        "description": "A tool that analyzes CT scans and classifies them for 18 different pathologies: Medical material, Arterial wall calcification, Cardiomegaly, Pericardial effusion, Coronary artery wall calcification, Hiatal hernia, Lymphadenopathy, Emphysema, Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Pleural effusion, Mosaic attenuation pattern, Peribronchial thickening, Consolidation, Bronchiectasis, Interlobular septal thickening. The output is a dictionary of pathologies and their predicted probabilities (0 to 1), the higher the probability the higher the likelihood that the pathology is present in the CT scan. Beware that this tool is not perfect and may produce false positives or false negatives.",
        "parameter": {
            "image_path": {
                "type": "string",
                "description": "The path of the input image.",
            },
        },
    },
    {
        "name": "effusion_segmentation_tool",
        "description": "A tool that segments pleural and pericardial effusions from CT scans using TotalSegmentator. The output is the list of file paths for the generated segmentation masks.",
        "parameter": {
            "image_path": {
                "type": "string",
                "description": "The path of the input CT image.",
            },
        },
    },
    {
        "name": "anatomy_segmentation_tool",
        "description": "A tool able to generate segmentation maps for various organs using TotalSegmentator. The output is the list of file paths for the generated segmentation masks.",
        "parameter": {
            "image_path": {
                "type": "string",
                "description": "The path of the input CT image.",
            },
            "organs": {
                "type": "array",
                "description": f'List of organs to segment from the CT image. Available organs are {(",").join(available_organs)}.',
            },
        },
    },
    {
        "name": "biggest_slice_selection_tool",
        "description": """Identifies the 2D axial slice from a CT volume that contains the largest area from the provided 3D segmentation map. If the lesion/organ spans multiple discontinuous regions, the tool selects one slice for each region, aiming to capture the most significant cross-sections. Informative slices are chosen by maximizing the intersection area of the segmented region within the chosen 2D planes. The selected slice(s) are saved as individual NumPy array files (e.g., .npy). The tool returns the list of file paths for the saved slices. If none of the slices in the CT contain the organ/lesion, the tool returns 'No relevant slices found in the segmentation map'.""",
        "parameter": {
            "image_path": {
                "type": "str",
                "description": "Path the to the input 3D CT image volume.",
            },
            "segmentation_path": {
                "type": "str",
                "description": "Path to the corresponding 3D segmentation map highlighting the organ or lesion of interest.",
            },
        },
    },
    {
        "name": "windowing_tool",
        "description": "A tool that applies CT window width/level adjustment to enhance the visualization of specific anatomical structures in CT images. The tool supports preset window types such as lung, bone, abdomen, and mediastinum. The output is a message indicating the location of the saved windowed image files. If the input file is NIfTI file input (.nii.gz) it returns a NIfTI file (.nii.gz). If the the input file is a 2D numpy array (.npy) it returns a PNG output.",
        "parameter": {
            "input_file": {
                "type": "array",
                "description": "List of file paths to the input files in .nii.gz or .npy format.",
            },
            "window": {
                "type": "string",
                "description": "Preset window type to apply. Options are 'lung', 'bone', 'abdomen', or 'mediastinum'.",
            },
        },
    },
    {
        "name": "get_several_slices_from_segmentation",
        "description": """Returns relevant 2D axial slices from a CT volume as highlighted by the provided segmentation map. The slices are chosen such thaht they span the entire volume highlighted by the segmentatation map. If the lesion/organ spans multiple discontinuous regions, the tool selects n_slices for each region, aiming to capture the most significant cross-sections. The selected slices are saved as individual NumPy array files (.npy). The tool returns the list of file paths for the saved slices. If none of the slices in the CT contain the organ/lesion, the tool returns 'No relevant slices found in the segmentation map'.""",
        "parameter": {
            "image_path": {
                "type": "str",
                "description": "Path the to the input 3D CT image volume.",
            },
            "segmentation_path": {
                "type": "str",
                "description": "Path to the corresponding 3D segmentation map highlighting the organ or lesion of interest.",
            },
            "n_slices": {
                "type": "int",
                "description": "Number of slices to extract from each discontinuous region. Default is 3.",
            },
        },
    },
    {
        "name": "slice_vqa_tool",
        "description": "A tool that can answer questions about 2D axial slices from CT images. The more precise the question the more accurate the tool answer is, for example asking 'is there cardiomegaly' might yield more accurate answers over simply asking 'is there any abnormality in this scan'. The tool can analyse multiple slices at once but can not analyse an entire CT volume.",
        "parameter": {
            "image_paths": {
                "type": "array",
                "description": "List of file paths to the input CT image slices in .png or .npy format.",
            },
            "question": {
                "type": "string",
                "description": "The question to be answered based on the content of the provided images.",
            },
        },
    },
    {
        "name": "report_judge_tool",
        "description": "A tool able to rate the quality of the generated report (only used to compute reward)",
        "parameter": {
            "ground_truth_report": {
                "type": "string",
                "description": "Ground truth report text.",
            },
            "candidate_report": {
                "type": "string",
                "description": "Candidate report text.",
            },
        },
    },
    {
        "name": "green_tool",
        "description": "A tool able to rate the quality of the generated report (only used to compute reward)",
        "parameter": {
            "ground_truth_report": {
                "type": "string",
                "description": "Ground truth report text.",
            },
            "candidate_report": {
                "type": "string",
                "description": "Candidate report text.",
            },
        },
    },
    {
        "name": "extract_slices_from_ct",
        "description": """Returns several 2D slices from a CT volume, spanning the entire volume. The selected slices are saved as individual NumPy array files (.npy). The tool returns the list of file paths for the saved slices.""",
        "parameter": {
            "image_path": {
                "type": "str",
                "description": "Path the to the input 3D CT image volume.",
            },
            "n_slices": {
                "type": "int",
                "description": "Number of slices to extract from the CT. Default is 5.",
            },
            "direction": {
                "type": "str",
                "description": "Direction to extract slices from. Options are 'axial', 'sagittal', or 'coronal'. Default is 'axial'.",
            },
        },
    },
]
