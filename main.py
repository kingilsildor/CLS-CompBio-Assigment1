from script.plot_data import plot_lineweaver_burk
from src.load_data import load_data


def main():
    file_path = "data/Kinetics.csv"
    data = load_data(file_path)
    plot_lineweaver_burk(data, save=True)


if __name__ == "__main__":
    main()
