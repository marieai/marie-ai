import os
import shutil

import torch
from transformers import pipeline

from marie.components.template_matching.document_matched import (
    load_template_matching_definitions,
    match_templates,
)
from marie.conf.helper import load_yaml
from marie.constants import __config_dir__, __model_path__
from marie.logging.mdc import MDC
from marie.logging.profile import TimeContext
from marie.pipe.extract_pipeline import ExtractPipeline, s3_asset_path, split_filename
from marie.storage import StorageManager
from marie.storage.s3_storage import S3StorageHandler
from marie.utils.docs import frames_from_file


def run_extract_pipeline():
    MDC.put("request_id", "test")

    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210327996/PID_1719_8755_0_210327996.tif"
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210329644/PID_3491_10833_0_210329644.tif"
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210329012/PID_457_6951_0_210329012.tif"
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210331524/PID_1756_8816_0_210331524.tif"
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210335376/PID_977_7770_0_210335376.tif"
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210335785/PID_5089_12292_0_210335785.tif"

    # 120791
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120791/Integration-120791-202930517/PID_1938_9366_0_202930517.tif"
    img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120791/Integration-120791-200687186/PID_3824_11135_0_200687186.tif"
    #
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210337854/PID_5140_12339_0_210337854.tif"
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210339152/PID_2015_9495_0_210339152.tif"
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210346394/PID_718_7393_0_210346394.tif"
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210346822/PID_1635_8610_0_210346822.tif"
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210350165/PID_3952_11235_0_210350165.tif"
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-120826/Integration-120826-210353682/PID_6948_14198_0_210353682.tif"

    # # 119662
    # img_path = "~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/Integration-119662/Integration-119662-210765089/PID_1699_8682_0_210765089.tif"

    root = os.path.expanduser("~/dev/workflow/mbx-grapnel/mbx-grapnel-engine/src/test/resources/test-deck/")
    img_path = "Integration-119662/Integration-119662-210767452/PID_1925_9282_0_210767452.tif"  # NO BOUNDARY
    img_path = "Integration-119662/Integration-119662-210983990/PID_5893_13226_0_210983990.tif"
    img_path = "Integration-119662/Integration-119662-210765105/PID_1805_9004_0_210765105.tif"
    img_path = "Integration-119662/Integration-119662-210766347/PID_885_7650_0_210766347.tif"
    img_path = "/Integration-119662/Integration-119662-210766868/PID_1692_8675_0_210766868.tif"
    img_path = "Integration-119662/Integration-119662-210766869/PID_1692_8675_0_210766869.tif"
    img_path = "Integration-119662/Integration-119662-210784432/PID_1807_9006_0_210784432.tif"
    img_path = "Integration-119662/Integration-119662-210784956/PID_269_6748_0_210784956.tif"
    img_path = "Integration-119662/Integration-119662-210765089/PID_1699_8682_0_210765089.tif"
    img_path = "Integration-119662/Integration-119662-211996043/PID_2034_9523_0_211996043.tif"
    img_path = "Integration-119662/Integration-119662-211994818/PID_7032_14288_0_211994818.tif"
    img_path = "Integration-119662/Integration-119662-211994322/PID_5942_13308_0_211994322.tif"
    img_path = "Integration-119662/Integration-119662-211972584/PID_893_7663_0_211972584.tif"
    img_path = "Integration-119662/Integration-119662-212217695/PID_2017_9499_0_212217695.tif"
    img_path = "Integration-119662/Integration-119662-212230067/PID_3973_11256_0_212230067.tif"
    img_path = "Integration-119662/Integration-119662-212288408/PID_2034_9523_0_212288408.tif"
    # img_path = "Integration-119662/Integration-119662-212028304/PID_3578_10901_0_212028304.tif"
    # img_path = "Integration-119662/Integration-119662-212021940/PID_7179_14435_0_212021940.tif"
    img_path = "Integration-115603/Integration-115603-212849075/PID_6145_13425_0_212849075.tif"
    img_path = "Integration-116486/Integration-116486-214202488/PID_2313_9784_0_214202488.tif"


    img_path = "Integration-115603/Integration-115603-214233146/PID_1315_8122_0_214233146.tif"

    img_path = os.path.join(root, img_path)
    img_path = os.path.join("/home/gbugaj/tmp/analysis/marie-issues/no-metadata/PID_7034_14289_0_215000470.tif")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found : {img_path}")

    # check if there is an .tif.original file and use that instead
    has_original = False
    if img_path.endswith(".tif"):
        img_path_alt = img_path + ".original"
        if os.path.exists(img_path_alt):
            has_original = True
            shutil.copy(img_path_alt, img_path)
            print(f"Replaced {img_path} with {img_path_alt}")

    filename, prefix, suffix = split_filename(img_path)
    print("Filename: ", filename)
    print("Prefix: ", prefix)
    print("Suffix: ", suffix)

    pipeline_config = load_yaml(os.path.join(__config_dir__, "tests-integration", "pipeline-integration.partial.yml"))
    pipeline = ExtractPipeline(pipeline_config=pipeline_config["pipeline"], cuda=True)
    has_boundary = True

    runtime_conf = {
        "template_matching": {
            "definition_id": "111539",  #  116486  115603 120826  120791 119662(PB)
        },
        "page_boundary": {
            "enabled": has_boundary,
        },
    }

    with TimeContext(f"### ExtractPipeline info"):
        # for k in range(0, 3):
        frames = frames_from_file(img_path)
        # frames = [frames[5]]
        results = pipeline.execute(ref_id=filename, ref_type="pid", frames=frames,
                                   runtime_conf=runtime_conf)
        print(results)
        work_dir = results["work_dir"]
        print("Work dir: ", work_dir)
        # get json filename from image_path
        filename = img_path.split("/")[-1].split(".")[0]
        filename = img_path.split("/")[-1]
        filename_no_ext = filename.split(".")[0]
        json_filename = os.path.join(work_dir, filename + ".meta.json")
        assets_dir = os.path.join(work_dir, "assets")
        print("Json filename: ", json_filename)

        if False:
            # # get the base dir from image_path and copy the json file to the base dir
            base_dir = os.path.dirname(img_path)
            print("Base dir: ", base_dir)
            shutil.copy(json_filename, base_dir)
            shutil.copytree(assets_dir, os.path.join(base_dir, "assets"))

            # copy individual items from assets to base dir for easy access
            shutil.copy(os.path.join(assets_dir, f"{filename_no_ext}.blobs.xml.zip"), base_dir)
            shutil.copy(os.path.join(assets_dir, f"{filename_no_ext}.pdf"), base_dir)
            shutil.copy(os.path.join(assets_dir, f"{filename_no_ext}.ocr.zip"), base_dir)
            shutil.copy(os.path.join(assets_dir, f"{filename}.clean"), base_dir)

            # if boundary enable make copy of the boundary image
            if not has_original and has_boundary:
                img_path_alt = img_path + ".original"
                shutil.copy(img_path, img_path_alt)

                boundary_img = os.path.join(assets_dir, filename)
                shutil.copy(boundary_img, base_dir)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["MARIE_DISABLE_CUDA"] = "True"
    torch.set_float32_matmul_precision('high')

    run_extract_pipeline()
