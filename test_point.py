from ransac_3d import RANSAC3D
from point import Point3D
import numpy as np

def main():
    data = np.random.random((100, 3))

    runsuck = RANSAC3D()
    runsuck.add_points(data)

    model = runsuck.fit(
            Point3D,
            10000,
            0.1
    )

    print('actual center: ', np.average(data, axis=0))
    print('model: ', model.get_model())
    

if __name__ == "__main__":
    main()
