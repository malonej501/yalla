#include <fstream>
#include <iostream>
#include <map>
#include <string>

#define COMPILE_AS_LIBRARY  // to avoid redefining main
#include "../examples/eggspot.cu"
#include "logging.cpp"

using namespace std;

// Simulation hyperparameters

const float mut_scale = 0.1;  // mutation scale
const int n_steps = 3;        // number of iterations in chain

string mutate(Pm& h_pm)
{
    // random mutation value
    float mut = static_cast<float>(rand()) / RAND_MAX * mut_scale;

    array<float*, 4> targets = {
        &h_pm.k_prod, &h_pm.k_deg, &h_pm.D_u, &h_pm.u_death};

    int target_idx = rand() % targets.size();  // pick random target


    bool is_negative = rand() % 2;  // give random sign to mutation
    if (is_negative) mut *= -1;

    const char* target_names[] = {"k_prod", "k_deg", "D_u", "u_death"};
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
        tissue_sim(0, NULL, i);
        // check output
        write_report_row(
            report_file, walk_id, i, attempt, status, target, h_pm);
    }
    return 0;
}