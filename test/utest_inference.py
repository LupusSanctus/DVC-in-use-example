import unittest
from inference import run_classifier
from pretrainedmodel.resnext101_32x4d import resnext101_32x4d

model = resnext101_32x4d(num_classes=1000,
                         pretrained='imagenet')

croco_data_path = "data/croco.jpg"
cat_data_path = "data/cat.jpg"

class_mapping = {"cat": "n02123159", "croco": "n01697457"}

class InferenceTest(unittest.TestCase):
    croco_max, croco_classname, croco_class_key = run_classifier(model, croco_data_path)
    cat_max, cat_classname, cat_class_key = run_classifier(model, cat_data_path)

    def test_crocoPredicted(self):
        self.assertEqual(self.croco_class_key, class_mapping.get("croco"))

    def test_catPredicted(self):
        self.assertEqual(self.cat_class_key, class_mapping.get("cat"))

    def test_crocoConfidencePassesThreshold(self):
        croco_confidence_threshold = 90
        croco_confidence = round(self.croco_max.item() * 100, 3)
        self.assertTrue(croco_confidence_threshold <= croco_confidence)

    def test_catConfidencePassesThreshold(self):
        cat_confidence_threshold = 90
        cat_confidence = round(self.cat_max.item() * 100, 3)
        self.assertTrue(cat_confidence_threshold <= cat_confidence)

if __name__ == '__main__':
    unittest.main()
