import unittest

from src.fluids.hydrogen_retrievers import (HydrogenRequesterFactory,
                                            HydrogenRetriever, PhaseRequester,
                                            SinglePhaseRequester,
                                            TwoPhaseRequester)


class TestSinglePhaseRequester(unittest.TestCase):

    def setUp(self) -> None:
        self.pressure = 300e5
        self.temperature = 70
        self.retriever = SinglePhaseRequester()

    def test_get_property(self):

        density = self.retriever.get_property(
            self.pressure, self.temperature, "D"
        )
        expected_density = 63.68658870474203
        self.assertEqual(expected_density, density)
    
    def test_get_hydrogen_properties(self):

        self.retriever.get_hydrogen_properties(
            self.pressure, self.temperature
        )


class TestTwoPhaseRequester(unittest.TestCase):

    def setUp(self) -> None:
        self.pressure = 0.090717e6
        self.retriever = TwoPhaseRequester()

    def test_get_property(self):
        temperature = self.retriever.get_property(
            self.pressure, "T", "gas"
        )
        expected_temperature = 20.0
        self.assertAlmostEqual(
            expected_temperature, temperature, places=4
        )

    def test_get_hydrogen_properties(self):

        self.retriever.get_hydrogen_properties(
            self.pressure, None
        )

        self.retriever.get_hydrogen_properties(
            None, 30
        )

    def test_compute_pressure_derivative(self):

        temp1 = 22
        self.assertEqual(
            self.retriever.compute_pressure_derivative(temp1),
            41319.799021746774
        )


class TestPhaseRequester(unittest.TestCase):

    def test_get_fluid_phase(self):

        # Supercritical test
        pressure = 300e5
        temperature = 70
        expected_value = "gas"
        actual_value = PhaseRequester().get_fluid_phase(
            temperature, pressure
        )
        self.assertEqual(expected_value, actual_value)

        # Supercritical gas test
        pressure = 3e5
        temperature = 1000
        expected_value = "gas"
        actual_value = PhaseRequester().get_fluid_phase(
            temperature, pressure
        )
        self.assertEqual(expected_value, actual_value)

        # Gas test
        pressure = 3e5
        temperature = 30
        expected_value = "gas"
        actual_value = PhaseRequester().get_fluid_phase(
            temperature, pressure
        )
        self.assertEqual(expected_value, actual_value)

        # Liquid test
        pressure = 3e5
        temperature = 22
        expected_value = "liquid"
        actual_value = PhaseRequester().get_fluid_phase(
            temperature, pressure
        )
        self.assertEqual(expected_value, actual_value)

        # Saturated test
        pressure = 8.04322035183e5
        temperature = 30.0000000
        expected_value = "twophase"
        actual_value = PhaseRequester().get_fluid_phase(
            temperature, pressure
        )
        self.assertEqual(expected_value, actual_value)


class TestHydrogenRequesterFactory(unittest.TestCase):

    def setUp(self) -> None:
        self.factory = HydrogenRequesterFactory()
    
    def test_get_hydrogen_retriever(self):
        
        retriever = self.factory.get_hydrogen_retriever("twophase")
        self.assertIsInstance(retriever, TwoPhaseRequester)

        retriever = self.factory.get_hydrogen_retriever("gas")
        self.assertIsInstance(retriever, SinglePhaseRequester)

        retriever = self.factory.get_hydrogen_retriever("liquid")
        self.assertIsInstance(retriever, SinglePhaseRequester)

        break_function = "break_something"
        with self.assertRaises(ValueError) as context:
            self.factory.get_hydrogen_retriever(break_function)
        exception  = f"'{break_function}' is an unsupported phase " \
            "for the hydrogen factory."
        self.assertTrue(exception in str(context.exception))


class TestHydrogenRetriever(unittest.TestCase):

    def setUp(self) -> None:
        self.retriever = HydrogenRetriever()

    def test_define_requester(self):

        with self.assertRaises(ValueError) as context:
            self.retriever.define_requester(None, None)
        exception  = "Not pressure nor temperature have been provided"
        self.assertTrue(exception in str(context.exception))

        pressure = 1.4e5
        temperature = None
        requester = self.retriever.define_requester(
            pressure, temperature
        )
        self.assertIsInstance(requester, TwoPhaseRequester)

        pressure = None
        temperature = 20
        requester = self.retriever.define_requester(
            pressure, temperature
        )
        self.assertIsInstance(requester, TwoPhaseRequester)

        pressure = 300e5
        temperature = 70
        requester = self.retriever.define_requester(
            pressure, temperature
        )
        self.assertIsInstance(requester, SinglePhaseRequester)

    def test_get_hydrogen_properties(self):
        pressure = 300e5
        temperature = 70
        hydrogen = self.retriever.get_hydrogen_properties(
            pressure, temperature
        )
        expected_value = pressure
        actual_value = hydrogen.pressure
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
