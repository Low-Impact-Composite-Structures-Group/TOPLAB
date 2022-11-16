

from analysis.recreate_rompokos_mission import perform_analysis


# def test_gas_draining():

#     # Define tank
#     tank = CylindricalTankSphericalCaps.rompokos()
    
#     # Define insulation
#     thickness = 4e-2
#     insulation = ConstantFoamInsulation.polyvinylchloride(thickness)

#     # Define fuel flow
#     fuel_flow = OutFlow.rompokos_cruise("gas")

#     # Define operating pressure window of the vessel
#     initial_pressures = [100e5, 200e5, 300e5, 400e5, 500e5, 600e5]
#     min_pressure = 10e5

#     data = list()
#     for initial_pressure in initial_pressures:
#         # Define initial state of tank
#         initial_temperature = 100
#         initial_fill = 0.0
#         initial_state = InitialState(
#             initial_pressure, initial_temperature, initial_fill
#         )

#         tank_states = gas_draining_analysis(
#             tank,
#             initial_state,
#             fuel_flow,
#             insulation,
#             min_pressure
#         )
#         data.append(tank_states)

#     fig1 = plot_tank_loads(
#         data,
#         [f'{int(pressure * 1e-5)} bar' for pressure in initial_pressures]
#     )
#     fig2 = plot_tank_temperatures(
#         data,
#         [f'{int(pressure * 1e-5)} bar' for pressure in initial_pressures]
#     )

#     fig3 = plot_tank_fill(
#         tank_states, 60
#     )

#     fig1.show()


# def test_liquid_draining():

#     # Define tank
#     tank = CylindricalTankSphericalCaps.rompokos()
    
#     # Define insulation
#     thickness = 4e-2
#     insulation = ConstantFoamInsulation.polyvinylchloride(thickness)

#     # Define fuel flow
#     fuel_flow = OutFlow.rompokos_cruise("liquid")

#     initial_pressures = [1.5e5, 2.0e5, 2.5e5, 3.0e5]

#     data = list()
#     for initial_pressure in initial_pressures:
#         # Define initial state of tank
#         initial_temperature = None
#         initial_fill = 0.95
#         initial_state = InitialState(
#             initial_pressure, initial_temperature, initial_fill
#         )

#         tank_states = liquid_draining_analysis(
#             tank,
#             initial_state,
#             fuel_flow,
#             insulation
#         )
#         data.append(tank_states)
#         print(max(tank_states, key=lambda x: x.pressure))
#     fig1 = plot_tank_loads(
#         data,
#         [f'{int(pressure * 1e-5)} bar' for pressure in initial_pressures]
#     )
#     fig2 = plot_tank_temperatures(
#         data,
#         [f'{int(pressure * 1e-5)} bar' for pressure in initial_pressures]
#     )

#     fig3 = plot_tank_fill(
#         tank_states, 60
#     )

#     fig1.show()


def main():
    perform_analysis()


if __name__ == "__main__":
    main()


# End
