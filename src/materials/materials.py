


from dataclasses import dataclass
from typing import Protocol


@dataclass
class Material:
    failure_stress: float
    density: float


@dataclass
class Metal(Material):

    def __post_init__(self):
        self.type = "metal"

    @classmethod
    def aluminum(cls):
        failure_stress = 450e6
        density = 2700
        return cls(failure_stress, density)


@dataclass
class Composite(Material):

    def __post_init__(self):
        self.type = "composite"



def main():
    pass


if __name__ == "__main__":
    main()


# End
