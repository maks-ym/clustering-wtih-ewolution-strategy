import data
import numpy as np

# TODO: split tests 1 test per assert statement
# TODO: move repeating constants out of functions

class TestHAPT:

    def test_get_train_data(self):
        d = data.HAPT()
        assert d._train_attrs is None
        d.get_train_data()
        assert len(d._train_attrs) > 0
        assert len(d.get_train_data()) > 0

    def test_get_train_labels(self):
        d = data.HAPT()
        assert d._train_labels is None
        d.get_train_labels()
        assert len(d._train_labels) > 0
        assert len(d.get_train_labels()) > 0

    def test_get_test_data(self):
        d = data.HAPT()
        assert d._test_attrs is None
        d.get_test_data()
        assert len(d._test_attrs) > 0
        assert len(d.get_test_data()) > 0

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
        assert len(d.get_train_data()) == len(d.get_train_labels())

    def test_load_test_data(self):
        d = data.HAPT()
        assert d._test_attrs is None
        assert d._test_labels is None
        d.load_test_data()
        assert len(d._test_attrs) > 0
        assert len(d._test_labels) > 0
        assert len(d._test_attrs) == len(d._test_labels)
        assert len(d.get_test_data()) == len(d.get_test_labels())

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
        assert len(d.get_train_data()) == len(d.get_train_labels())
        assert len(d.get_test_data()) == len(d.get_test_labels())

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

    def test_aggregate_groups(self):
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
        d._labels = orig_labels
        d._test_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        d._train_labels = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        d.aggregate_groups()
        assert np.array_equal(d._aggregated_test_labels, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]))
        assert np.array_equal(d._aggregated_train_labels, np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]))
        assert d._aggregated2initial_labels == {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9, 10, 11, 12]}

    def test_get_aggr2initial_labs_map(self):
        d = data.HAPT()
        d.load_all_data()
        d.aggregate_groups()
        assert d.get_aggr2initial_labs_map() == {
            'WALKING': ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'],
            'STATIC': ['SITTING', 'STANDING', 'LAYING'],
            'TRANSITION': ['STAND_TO_SIT', 'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']
        }

    def test_get_aggregated_test_labels(self):
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
        d._labels = orig_labels
        d._test_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        d._train_labels = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        assert d.get_aggregated_test_labels() == d._test_labels
        d.aggregate_groups()
        print(d._aggregated_test_labels)
        assert np.array_equal(d.get_aggregated_test_labels(), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]))

    def test_get_aggregated_train_labels(self):
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
        d._labels = orig_labels
        d._test_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        d._train_labels = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        assert d.get_aggregated_train_labels() == d._train_labels
        d.aggregate_groups()
        assert np.array_equal(d.get_aggregated_train_labels(), np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0]))

    def test_get_aggregated_labels_map(self):
        d = data.HAPT()
        assert d.get_aggregated_labels_map() == {0: "WALKING", 1: "STATIC", 2: "TRANSITION"}
