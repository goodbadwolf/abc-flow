#include <cstring>
#include <iostream>
#include <random>
#include <string>

#include "ftle.h"

void printUsage(const char *programName)
{
    std::cout << "Usage: " << programName << " [options]\n"
              << "Options:\n"
              << "  --res N        Grid resolution (default: 64)\n"
              << "  --format TYPE  Output format: vtk or vdb (default: vtk)\n"
              << "  --A VALUE      Value for A in ABC flow (default: 1.0)\n"
              << "  --B VALUE      Value for B in ABC flow (default: 1.0)\n"
              << "  --C VALUE      Value for C in ABC flow (default: 1.0)\n"
              << "  --randomize    Randomize the flow parameters\n"
              << "  --numIso N     Number of isosurfaces to generate (default: 5)\n"
              << "  --help         Show this help message\n";
}

int main(int argc, char *argv[])
{
    int resolution = 64;
    std::string format = "vtk";
    double A = 1.7320;
    double B = 1.4142;
    double C = 1.0;
    bool randomize = false;
    int numIso = 5;

    bool providedA = false;
    bool providedB = false;
    bool providedC = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--res") == 0 && i + 1 < argc)
        {
            resolution = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc)
        {
            format = argv[++i];
            if (format != "vtk" && format != "vdb")
            {
                std::cerr << "Error: format must be either 'vtk' or 'vdb'\n";
                return 1;
            }
        }
        else if (strcmp(argv[i], "--A") == 0 && i + 1 < argc)
        {
            A = std::stod(argv[++i]);
            providedA = true;
        }
        else if (strcmp(argv[i], "--B") == 0 && i + 1 < argc)
        {
            B = std::stod(argv[++i]);
            providedB = true;
        }
        else if (strcmp(argv[i], "--C") == 0 && i + 1 < argc)
        {
            C = std::stod(argv[++i]);
            providedC = true;
        }
        else if (strcmp(argv[i], "--randomize") == 0)
        {
            randomize = true;
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    FTLEComputer ftle(resolution, format);
    if (randomize)
    {
        const double A_MIN = 0.5, A_MAX = 2.0;
        const double B_MIN = 0.5, B_MAX = 2.0;
        const double C_MIN = 0.5, C_MAX = 2.0;

        std::random_device rd;
        std::mt19937 gen(rd());
        if (!providedA)
        {
            std::uniform_real_distribution<double> distA(A_MIN, A_MAX);
            A = distA(gen);
        }
        if (!providedB)
        {
            std::uniform_real_distribution<double> distB(B_MIN, B_MAX);
            B = distB(gen);
        }
        if (!providedC)
        {
            std::uniform_real_distribution<double> distC(C_MIN, C_MAX);
            C = distC(gen);
        }
    }

    std::cout << "Setting ABC flow parameters: A=" << A << ", B=" << B << ", C=" << C << "\n";
    ftle.setABCParameters(A, B, C);

    std::cout << "Computing FTLE...\n";
    ftle.computeFTLE();

    std::cout << "Rendering FTLE. Press 'S' to save the output during rendering using format " << format << "\n";
    ftle.renderWithSave();

    return 0;
}