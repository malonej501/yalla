#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define COMPILE_AS_LIBRARY  // to avoid redefining main
// #include "../examples/eggspot.cu"
// #include "../examples/volk.cu"
#include "../examples/eggspot_layers.cu"
#include "logging.cpp"  // for write_report_header/row

using namespace std;

// Simulation hyperparameters

// Random walk
const float mut_mplr = 10;  // mutation multiplier
const int n_steps = 1000;   // number of iterations in chain

// Latin hypercube

// Save initial parameter values
Pm h_def_pm;

// Per-parameter integer ranges for LHS (order: iil, xid, ixl, xxl, xidd, ixdd)
static array<int, 6> lh_low = {1, 1, 1, 1, 1, 1};
static array<int, 6> lh_high = {100, 100, 100, 100, 100, 100};

// Setter to allow customizing ranges before sampling
void set_lh_ranges(const array<int, 6>& low, const array<int, 6>& high)
{
    for (int i = 0; i < 6; ++i) {
        lh_low[i] = low[i];
        lh_high[i] = high[i];
        if (lh_high[i] <= lh_low[i]) lh_high[i] = lh_low[i] + 1;
    }
}

string mutate(Pm& h_pm)
{
    array<int*, 6> targets = {
        &h_pm.iil, &h_pm.xid, &h_pm.ixl, &h_pm.xxl, &h_pm.xidd, &h_pm.ixdd};
    int target_idx = rand() % targets.size();  // pick random target

    // float mut_scale = mut_mplr;
    // random mutation value
    float mut = static_cast<float>(rand()) / RAND_MAX * mut_mplr;
    int mut_int = static_cast<int>(round(mut));

    bool is_negative = rand() % 2;  // give random sign to mutation
    if (is_negative) {
        mut *= -1;
        mut_int *= -1;
    }

    const char* target_names[] = {"iil", "xid", "ixl", "xxd", "xidd", "ixdd"};
    cout << "\n\nMutating " << target_names[target_idx] << " by " << mut
         << "\n\n"
         << endl;

    *targets[target_idx] += mut_int;

    return target_names[target_idx];
}

string mutate_lh(Pm& h_pm)
{
    // Latin Hypercube mutation: sets all six integer targets according to
    // a precomputed Latin Hypercube sample. On first call the LHS matrix
    // is generated using per-parameter ranges stored in `lh_low`/`lh_high`.
    // N.B. integer collisisons may occur due to rounding.

    static bool initialized = false;  // whether LHS samples have been generated
    static vector<array<int, 6>> samples;  // n_steps rows x 6 params
    static int next_row = 0;

    if (!initialized) {  // construct samples only once
        const int n = n_steps > 0 ? n_steps : 1;
        samples.resize(n);

        // RNG (use fixed seed for reproducibility if desired)
        random_device rd;
        mt19937 gen(rd());

        for (int j = 0; j < 6; ++j) {  // for each parameter
            int low = lh_low[j];
            int high = lh_high[j];
            int range = high - low;
            if (range <= 0) range = 1;

            // generate stratified points and random offset within strata
            vector<double> points(n);      // no. points = no. samples
            for (int i = 0; i < n; ++i) {  // for each stratum
                uniform_real_distribution<> d(0.0, 1.0);
                points[i] = (i + d(gen)) / static_cast<double>(n);
            }

            // scale to integer range
            vector<int> col(n);
            for (int i = 0; i < n; ++i) {
                double scaled = low + points[i] * range;
                col[i] = static_cast<int>(round(scaled));
            }

            // permute column to create LHS
            // previous loop creates samples based on value of i, so shuffle to
            // remove correlation between parameters (i.e. so not along diagonal
            // of hypercube)
            shuffle(col.begin(), col.end(), gen);

            // write into samples
            for (int i = 0; i < n; ++i) samples[i][j] = col[i];
            cout << "LHS parameter " << j << " range [" << low << ", " << high
                 << "], samples: ";
        }

        initialized = true;
        next_row = 0;
    }

    int row = next_row % static_cast<int>(samples.size());
    auto& vals = samples[row];

    h_pm.iil = vals[0];
    h_pm.xid = vals[1];
    h_pm.ixl = vals[2];
    h_pm.xxl = vals[3];
    h_pm.xidd = vals[4];
    h_pm.ixdd = vals[5];

    next_row++;

    const char* names[] = {"iil", "xid", "ixl", "xxl", "xidd", "ixdd"};
    cout << "Applied LHS row " << row << ": ";
    for (int k = 0; k < 6; ++k)
        cout << names[k] << "=" << vals[k] << (k + 1 < 6 ? ", " : "\n");

    return string("mutate_lh_row_" + to_string(row));
}

// string mutate(Pm& h_pm)
// {
//     // random mutation value
//     float mut = static_cast<float>(rand()) / RAND_MAX * mut_scale;
//     bool is_negative = rand() % 2;  // give random sign to mutation
//     if (is_negative) mut *= -1;

//     map<string, auto> targets = {{"k_prod", &h_pm.k_prod},
//         {"k_deg", &h_pm.k_deg}, {"D_u", &h_pm.D_u}, {"u_death",
//         &h_pm.u_death}};

//     // Choose a random pair from the map
//     auto t_pair = targets.begin();
//     advance(t_pair, rand() % targets.size());

//     cout << "\n\nMutating " << t_pair.first << " by " << mut << "\n\n" <<
//     endl;

//     t_pair.second += mut;

//     return t_pair.first;
// }

int main()
{
    int walk_id = 0;
    int attempt = 0;
    string status = "pass";

    ofstream report_file(  // begin report file
        "../run/output/report_" + to_string(walk_id) + ".csv");
    write_report_header(report_file);

    for (int i = 0; i < n_steps; i++) {
        cout << "Step: " << i << endl;
        string target = mutate_lh(h_pm);  // select target and change
        tissue_sim(0, NULL, walk_id, i);
        // check output

        system(
            (". ../venv/bin/activate && python3 ../run/render.py output -f 2 "
             "-w " +
                to_string(walk_id) + " -s " + to_string(i))
                .c_str());
        write_report_row(
            report_file, walk_id, i, attempt, status, target, h_pm);
        report_file.flush();  // flush the buffer
    }
    return 0;
}