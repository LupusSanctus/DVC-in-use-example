import unittest

from inference import run_classifier
from pretrainedmodel.resnext101_32x4d import resnext101_32x4d

model = resnext101_32x4d(num_classes=1000,
                         pretrained='imagenet')

croco_data_path = "data/croco.jpg"
tiger_data_path = "data/tiger.jpg"

cat_data_path = "data/cat.jpg"
persian_cat_data_path = "data/persian_cat.jpg"
tabby_cat_data_path = "data/tabby_cat.jpeg"
egyptian_mau_data_path = "data/egyptian_cat.jpg"

class_mapping = {"tiger": "n02129604",
                 "croco": "n01697457",
                 "cat": "n02123159",
                 "persian_cat": "n02123394",
                 "tabby_cat": "n02123045",
                 "egyptian_cat": "n02124075"}


class InferenceTest(unittest.TestCase):
    croco_max, croco_classname, croco_class_key = run_classifier(model, croco_data_path)
    cat_max, cat_classname, cat_class_key = run_classifier(model, cat_data_path)
    pcat_max, pcat_classname, pcat_class_key = run_classifier(model, persian_cat_data_path)
    confidence_threshold = 90

    def test_crocoPredicted(self):
        self.assertEqual(self.croco_class_key, class_mapping.get("croco"))

    def test_catClassPredicted(self):
        self.assertTrue(self.cat_class_key is not class_mapping.get("croco"))

    def test_catPredicted(self):
        self.assertEqual(self.cat_class_key, class_mapping.get("cat"))

    def test_persianCatPredicted(self):
        self.assertEqual(self.pcat_class_key, class_mapping.get("persian_cat"))

    def test_tabbyCatPredicted(self):
        cat_max, cat_classname, cat_class_key = run_classifier(model, tabby_cat_data_path)
        self.assertEqual(cat_class_key, class_mapping.get("tabby_cat"))

    def test_egyptianCatPredicted(self):
        cat_max, cat_classname, cat_class_key = run_classifier(model, egyptian_mau_data_path)
        self.assertEqual(cat_class_key, class_mapping.get("egyptian_cat"))

    def test_crocoConfidencePassesThreshold(self):
        croco_confidence = round(self.croco_max.item() * 100, 3)
        self.assertTrue(self.confidence_threshold <= croco_confidence)

    def test_persianCatConfidencePassesThreshold(self):
        pcat_confidence = round(self.pcat_max.item() * 100, 3)
        self.assertTrue(self.confidence_threshold <= pcat_confidence)


if __name__ == '__main__':
    unittest.main()
