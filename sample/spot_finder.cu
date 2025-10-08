#include <fstream>
#include <iostream>
#include <string>

#define COMPILE_AS_LIBRARY  // to avoid redefining main
#include "../examples/eggspot.cu"
#include "logging.cpp"

using namespace std;

// Simulation hyperparameters

const float mut_mplr = 0.1;  // mutation multiplier
const int n_steps = 1000;    // number of iterations in chain

// Save initial parameter values
Pm h_def_pm;

string mutate(Pm& h_pm)
{
    array<float*, 5> array<float*, 5> targets = {
        &h_pm.A_div, &h_pm.k_prod, &h_pm.k_deg, &h_pm.D_u, &h_pm.u_death};
    int target_idx = rand() % targets.size();  // pick random target

    float mut_scale = mut_mplr
        // random mutation value
        float mut = static_cast<float>(rand()) / RAND_MAX * mut_scale;


    bool is_negative = rand() % 2;  // give random sign to mutation
    if (is_negative) mut *= -1;

    const char* target_names[] = {"A_div", "k_prod", "k_deg", "D_u", "u_death"};
    cout << "\n\nMutating " << target_names[target_idx] << " by " << mut
         << "\n\n"
         << endl;

    *targets[target_idx] += mut;

    return target_names[target_idx];
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
//     std::advance(t_pair, rand() % targets.size());

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
        "../run/output/report_" + std::to_string(walk_id) + ".csv");
    write_report_header(report_file);

    for (int i = 0; i < n_steps; i++) {
        cout << "Step: " << i << endl;
        string target = mutate(h_pm);  // select target and change
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