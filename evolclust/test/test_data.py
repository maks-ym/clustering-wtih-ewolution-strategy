import data

class TestHAPT:

    def test_get_train_feats(self):
        d = data.HAPT()
        assert d._train_attrs is None
        d.get_train_feats()
        assert len(d._train_attrs) > 0
        assert len(d.get_train_feats()) > 0

    def test_get_train_labels(self):
        d = data.HAPT()
        assert d._train_labels is None
        d.get_train_labels()
        assert len(d._train_labels) > 0
        assert len(d.get_train_labels()) > 0

    def test_get_test_feats(self):
        d = data.HAPT()
        assert d._test_attrs is None
        d.get_test_feats()
        assert len(d._test_attrs) > 0
        assert len(d.get_test_feats()) > 0

    def test_get_test_labels(self):
        d = data.HAPT()
        assert d._test_labels is None
        d.get_test_labels()
        assert len(d._test_labels) > 0
        assert len(d.get_test_labels()) > 0

    def test_load_train_data(self):
        d = data.HAPT()
        assert d._train_attrs is None
        assert d._train_labels is None
        d.load_train_data()
        assert len(d._train_attrs) > 0
        assert len(d._train_labels) > 0
        assert len(d._train_attrs) == len(d._train_labels)
        assert len(d.get_train_feats()) == len(d.get_train_labels())

    def test_load_test_data(self):
        d = data.HAPT()
        assert d._test_attrs is None
        assert d._test_labels is None
        d.load_test_data()
        assert len(d._test_attrs) > 0
        assert len(d._test_labels) > 0
        assert len(d._test_attrs) == len(d._test_labels)
        assert len(d.get_test_feats()) == len(d.get_test_labels())

    def test_load_all_data(self):
        d = data.HAPT()
        assert d._train_attrs is None
        assert d._train_labels is None
        assert d._test_attrs is None
        assert d._test_labels is None
        d.load_all_data()
        assert len(d._train_attrs) > 0
        assert len(d._train_labels) > 0
        assert len(d._test_attrs) > 0
        assert len(d._test_labels) > 0
        assert len(d._train_attrs) == len(d._train_labels)
        assert len(d._test_attrs) == len(d._test_labels)
        assert len(d.get_train_feats()) == len(d.get_train_labels())
        assert len(d.get_test_feats()) == len(d.get_test_labels())

    def test_get_labels_map(self):
        orig_labels = {
            1: "WALKING",
            2: "WALKING_UPSTAIRS",
            3: "WALKING_DOWNSTAIRS",
            4: "SITTING",
            5: "STANDING",
            6: "LAYING",
            7: "STAND_TO_SIT",
            8: "SIT_TO_STAND",
            9: "SIT_TO_LIE",
            10: "LIE_TO_SIT",
            11: "STAND_TO_LIE",
            12: "LIE_TO_STAND"
        }
        d = data.HAPT()
        assert d._labels == {}
        d.get_labels_map()
        assert d._labels == orig_labels
        assert d.get_labels_map() == orig_labels
