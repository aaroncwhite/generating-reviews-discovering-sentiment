from unittest import TestCase
from .encoder import Model

class TestNeuronModel(TestCase):
    """Test that the model can load and analyze
    sentiment on texts
    """

    # Since the source data mainly were product reviews...
    test_phrases = (('I loved it!', 1.076),
                    ('This sucks.', -1.3407),
                    ('I will definitely come back to this establishment!', .8168))

    def setUp(self):
        self.model = Model()

    def test_transform(self):
        for test in self.test_phrases:
            with self.subTest(test=test):
                result = self.model.predict(test[0])
                self.assertAlmostEqual(result[0].sentiment, test[1])

