import os
import glob
import yaml


class Captcha(object):
    def __init__(self, config):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(**config)

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        result = self.ocr.predict(input=im_path)
        for res in result:
            res.save_to_json(
                os.path.join(
                    save_path,
                    os.path.basename(im_path)
                    .replace(".jpg", ".json")
                    .replace("input", "output"),
                )
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--im_directory",
        type=str,
        default="./sampleCaptchas/input/",
        help="captcha image directory",
    )
    parser.add_argument(
        "--save_path", type=str, default="./results", help="path to save the output"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to config file"
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    captcha_files = glob.glob(os.path.join(args.im_directory, "*.jpg"))

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Model config : ", config)
    model = Captcha(config=config)
    for im_path in captcha_files:
        print("inferring on ", im_path)
        model(im_path, args.save_path)
