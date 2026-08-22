class ThermalManager:
    """
    Handles thermal safety calculations for the MEC system.
    """

    def __init__(self, thermal_threshold=85.0):
        self.thermal_threshold = thermal_threshold

    def calculate_headroom(self, predicted_temperature):
        """
        Calculate remaining safe temperature capacity.
        """

        headroom = self.thermal_threshold - predicted_temperature

        return headroom

    def get_thermal_status(self, predicted_temperature):
        """
        Determine the thermal condition of the system.
        """

        if predicted_temperature < 60:
            return "COOL"

        elif predicted_temperature < 75:
            return "NORMAL"

        elif predicted_temperature < self.thermal_threshold:
            return "HIGH"

        else:
            return "CRITICAL"

    def is_thermal_safe(self, predicted_temperature):
        """
        Check whether the temperature is below the safety threshold.
        """

        return predicted_temperature < self.thermal_threshold


# Testing the Thermal Manager
if __name__ == "__main__":

    thermal_manager = ThermalManager(
        thermal_threshold=85.0
    )

    predicted_temperature = 81.73

    headroom = thermal_manager.calculate_headroom(
        predicted_temperature
    )

    status = thermal_manager.get_thermal_status(
        predicted_temperature
    )

    safe = thermal_manager.is_thermal_safe(
        predicted_temperature
    )

    print("=" * 60)
    print("THERMAL MANAGEMENT SYSTEM")
    print("=" * 60)

    print(f"Predicted Temperature: {predicted_temperature:.2f} °C")
    print(f"Thermal Threshold:    85.00 °C")
    print(f"Thermal Headroom:     {headroom:.2f} °C")
    print(f"Thermal Status:       {status}")
    print(f"Thermal Safe:         {safe}")

    print("=" * 60)