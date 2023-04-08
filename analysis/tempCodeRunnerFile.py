    # performances = [
    #     [
    #         DrainingAnalysisFacade.analyse(
    #             TankDimensions(
    #                 radius, body_length
    #             ),
    #             tank_material,
    #             insulation,
    #             fuel_mass_flow,
    #             fuel_flow_phase,
    #             initial_state,
    #             OperatingEnvelope(
    #                 None,
    #                 min_pressure,
    #                 None
    #             )
    #         )
    #         for body_length in body_lengths
    #     ]
    #     for radius in radii
    # ]

    # save_results(radii, body_lengths, performances, directory)