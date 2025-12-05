import csv
from pathlib import Path

# Local import of NIST aluminum model
from nist_properties.aluminum_6061T6_nist import specific_heat


def main():
    base_dir = Path(__file__).parent
    output_csv = base_dir / "aluminum_6061T6_cps.csv"

    # Generate temperatures from 4 K to 300 K inclusive
    temps = list(range(4, 301))

    # Compute aluminum Cp at these temperatures
    rows = [("Temp [K]", "Specific heat capacity [J/kg.K]")]
    for T in temps:
        cp = specific_heat(T)
        rows.append((f"{T:g}", f"{cp:.9f}"))

    # Write output CSV with identical header format
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
