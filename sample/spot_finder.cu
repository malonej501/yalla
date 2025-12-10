#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <Python.h>

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
// static array<int, 6> lh_low = {1, 1, 1, 1, 1, 1};
// static array<int, 6> lh_high = {100, 100, 100, 100, 100, 100};
template<typename T>
struct Targets {
    const char* name;
    T low;
    T high;
    T Pm::* member;  // pointer to member of Pm
};

static constexpr std::array<Targets<int>, 6> lh_int_targets{
    {{"iil", 10, 30, &Pm::iil}, 
    {"xid", 30, 50, &Pm::xid},
    {"ixl", 1, 10, &Pm::ixl}, 
    {"xxl", 30, 50, &Pm::xxl},
    {"xidd", 20, 40, &Pm::xidd}, 
    {"ixdd", 1, 10, &Pm::ixdd}}
};

static constexpr std::array<Targets<float>, 4> lh_float_targets{
    {{"qia", 0.001f, 0.004f, &Pm::qia}, 
    {"qxa", 0.01f, 0.04f, &Pm::qxa},
    {"qir", 0.001f, 0.01f, &Pm::qir}, 
    {"qxr", 0.001f, 0.1f, &Pm::qxr}}
};

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
    static bool initialized = false;
    static vector<vector<double>>
        samples;  // each row is all parameters (ints+floats)
    static int next_row = 0;

    const int n_int = lh_int_targets.size();
    const int n_float = lh_float_targets.size();
    const int n_params = n_int + n_float;

    if (!initialized) {

        samples.assign(n_steps, vector<double>(n_params));

        std::random_device rd;
        std::mt19937 gen(rd());

        // ----------------------------------------
        // Generate LHS columns for INT parameters
        // ----------------------------------------
        for (int j = 0; j < n_int; ++j) {
            const auto& tgt = lh_int_targets[j];
            double low = tgt.low;
            double high = tgt.high;
            double range = std::max(1.0, high - low);

            // stratified points
            vector<double> pts(n_steps);
            std::uniform_real_distribution<> dist(0.0, 1.0);
            
            for (int i = 0; i < n_steps; ++i) pts[i] = (i + dist(gen)) / n_steps;

            // scale to range
            vector<double> col(n_steps);
            for (int i = 0; i < n_steps; ++i) col[i] = low + pts[i] * range;
            // permute the column
            std::shuffle(col.begin(), col.end(), gen);

            for (int i = 0; i < n_steps; ++i) samples[i][j] = col[i];
            cout << "LHS int param " << tgt.name << " range [" << low << ", "
                 << high << "]\n";
        }

        // ----------------------------------------
        // Generate LHS columns for FLOAT parameters
        // ----------------------------------------
        for (int j = 0; j < n_float; ++j) {
            const auto& tgt = lh_float_targets[j];
            double low = tgt.low;
            double high = tgt.high;
            double range = std::max(1e-12, high - low);

            vector<double> pts(n_steps);
            std::uniform_real_distribution<> dist(0.0, 1.0);

            for (int i = 0; i < n_steps; ++i) pts[i] = (i + dist(gen)) / n_steps;
            vector<double> col(n_steps);
            for (int i = 0; i < n_steps; ++i) col[i] = low + pts[i] * range;

            std::shuffle(col.begin(), col.end(), gen);

            for (int i = 0; i < n_steps; ++i) samples[i][n_int + j] = col[i];
            cout << "LHS float param " << tgt.name << " range [" << low << ", "
                 << high << "]\n";
        }

        initialized = true;
        next_row = 0;
    }

    // ----------------------------------------
    // Apply one row of samples to Pm
    // ----------------------------------------
    int row = next_row % samples.size();
    const auto& vals = samples[row];

    // ints
    for (size_t j = 0; j < lh_int_targets.size(); ++j) {
        const auto& tgt = lh_int_targets[j];
        h_pm.*(tgt.member) = int(std::round(vals[j]));
    }

    // floats
    for (size_t j = 0; j < lh_float_targets.size(); ++j) {
        const auto& tgt = lh_float_targets[j];
        h_pm.*(tgt.member) = float(vals[n_int + j]);
    }

    next_row++;

    // printing (optional)
    cout << "Applied LHS row " << row << ": ";
    for (int j = 0; j < n_int; ++j)
        cout << lh_int_targets[j].name << "=" << int(std::round(vals[j]))
             << ", ";
    for (int j = 0; j < n_float; ++j)
        cout << lh_float_targets[j].name << "=" << vals[n_int + j]
             << (j + 1 < n_float ? ", " : "\n");

    return "mutate_lh_row_" + to_string(row);
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

// int main(int argc, char const* argv[])
// {
//     string out_dir_name = "output";
//     if (argc > 1) out_dir_name = string(argv[1]);
//     int walk_id = 0;
//     int attempt = 0;
//     string status = "pass";
//     Py_Initialize();

//     ofstream report_file(  // begin report file
//         "../run/" + out_dir_name + "/report_" + to_string(walk_id) + ".csv");
//     write_report_header(report_file);

//     for (int i = 0; i < n_steps; i++) {
//         cout << "Step: " << i << endl;
//         string target = mutate_lh(h_pm);  // select target and change
//         tissue_sim(0, NULL, walk_id, i, true, out_dir_name);
//         // check output
        
//         system(
//             (". ../venv/bin/activate && python3 ../run/render.py " + 
//                 out_dir_name + " -f 2 -w " + to_string(walk_id) + 
//                 " -s " + to_string(i)).c_str());
//         write_report_row(
//             report_file, walk_id, i, attempt, status, target, h_pm);
//         report_file.flush();  // flush the buffer
//     }
//     return 0;
// }

int main(int argc, char const* argv[])
{
    string out_dir_name = "output";
    if (argc > 1) out_dir_name = string(argv[1]);
    int walk_id = 0;
    int attempt = 0;
    string status = "pass";

    // Set PythonHome to venv
    setenv("PYTHONPATH", "../venv/lib/python3.11/site-packages", 1);

    // Load python interpreter for phenotyping
    Py_Initialize();
    PyRun_SimpleString("import sys; sys.path.append('../run')");
    PyObject* pName = PyUnicode_FromString("render");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to load render module\n";
        return 1;
    }

    // Get pattern_stats_frame function
    PyObject* pStats = PyObject_GetAttrString(pModule, "pattern_stats_frame");
    if (!pStats || !PyCallable_Check(pStats)) {
        PyErr_Print();
        std::cerr << "Cannot find function pattern_stats_frame\n";
        Py_XDECREF(pStats);
        Py_DECREF(pModule);
        return 1;
    }
    // Get render_frame function
    PyObject* pRender = PyObject_GetAttrString(pModule, "render_frame");
    if (!pRender || !PyCallable_Check(pRender)) {
        PyErr_Print();
        std::cerr << "Cannot find function render_frame\n";
        Py_XDECREF(pRender);
        Py_XDECREF(pStats);
        Py_DECREF(pModule);
        return 1;
    }

    ofstream report_file(  // begin report file
        "../run/" + out_dir_name + "/report_" + to_string(walk_id) + ".csv");
    write_report_header(report_file);

    for (int i = 0; i < n_steps; i++) {
        cout << "Step: " << i << endl;
        string target = mutate_lh(h_pm);  // select target and change
        tissue_sim(0, NULL, walk_id, i, true, out_dir_name);

        // Declare metrics before the Python call
        int n_clusters = 0;
        double mean_area = 0.0;
        double mean_roundness = 0.0;
        
        // Call pattern_stats_frame with the right arguments
        // Assuming signature: pattern_stats_frame(folder, walk_id, step, )
        PyObject* pArgs = PyTuple_Pack(3,
            PyUnicode_FromString(out_dir_name.c_str()),
            PyLong_FromLong(walk_id),
            PyLong_FromLong(i));  
        
        PyObject* pResult = PyObject_CallObject(pStats, pArgs);
        PyObject* pRenderResult = PyObject_CallObject(pRender, pArgs);
        Py_DECREF(pArgs);

        
        if (pResult) {
            // Check if result is a tuple with 3 elements
            if (PyTuple_Check(pResult) && PyTuple_Size(pResult) == 3) {
                PyObject* pItem0 = PyTuple_GetItem(pResult, 0);  // borrowed ref
                PyObject* pItem1 = PyTuple_GetItem(pResult, 1);  // borrowed ref
                PyObject* pItem2 = PyTuple_GetItem(pResult, 2);  // borrowed ref
                
                n_clusters = PyLong_AsLong(pItem0);
                mean_area = PyFloat_AsDouble(pItem1);
                mean_roundness = PyFloat_AsDouble(pItem2);
                
                cout << "Pattern metrics: " << n_clusters << ", " << mean_area << ", " << mean_roundness << endl;
                // Use mean_area, mean_roundness in your logic (store in report, etc.)
            } else {
                std::cerr << "Expected tuple of 2 floats from pattern_stats_frame\n";
            }
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
            std::cerr << "Python function call failed\n";
        }
        
        write_report_row(
            report_file, walk_id, i, attempt, status, target, 
            n_clusters, mean_area, mean_roundness, h_pm);
        report_file.flush();  // flush the buffer
    }

    // Cleanup
    Py_XDECREF(pRender);
    Py_XDECREF(pStats);
    Py_DECREF(pModule);
    Py_Finalize();
    return 0;
}