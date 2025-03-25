#include <fstream>
#include <string>
#include "../params/params.h"

void write_report_header(std::ofstream& file) {
	file << "walk_id, step, attempt, status, target, r_max, n_max, noise, cont_time, dt, no_frames, tmode, init_dist, div_dist, n_0, A_init, tis_s, fin_walls, w_off_s, ray_switch, n_rays, s_ray, mov_switch, diff_adh_rep, rii, Rii, aii, Aii, rdd, Rdd, add, Add, adv_switch, ad_s, soft_ad_s, prolif_switch, pmode, A_div, B_div, C_div, r_A_birth, uthresh, vthresh, mech_thresh, cmode, k_prod, k_deg, D_u, D_v, a_u, b_v, type_switch, death_switch, u_death\n";
}

void write_report_row(std::ofstream& file, int walk_id, int i, int attempt, std::string status, std::string target, const Pm& h_pm) {
	file  << walk_id << "," << i << "," << attempt << "," << status << "," << target << "," << h_pm.r_max << "," << h_pm.n_max << "," << h_pm.noise << "," << h_pm.cont_time << "," << h_pm.dt << "," << h_pm.no_frames << "," << h_pm.tmode << "," << h_pm.init_dist << "," << h_pm.div_dist << "," << h_pm.n_0 << "," << h_pm.A_init << "," << h_pm.tis_s << "," << h_pm.fin_walls << "," << h_pm.w_off_s << "," << h_pm.ray_switch << "," << h_pm.n_rays << "," << h_pm.s_ray << "," << h_pm.mov_switch << "," << h_pm.diff_adh_rep << "," << h_pm.rii << "," << h_pm.Rii << "," << h_pm.aii << "," << h_pm.Aii << "," << h_pm.rdd << "," << h_pm.Rdd << "," << h_pm.add << "," << h_pm.Add << "," << h_pm.adv_switch << "," << h_pm.ad_s << "," << h_pm.soft_ad_s << "," << h_pm.prolif_switch << "," << h_pm.pmode << "," << h_pm.A_div << "," << h_pm.B_div << "," << h_pm.C_div << "," << h_pm.r_A_birth << "," << h_pm.uthresh << "," << h_pm.vthresh << "," << h_pm.mech_thresh << "," << h_pm.cmode << "," << h_pm.k_prod << "," << h_pm.k_deg << "," << h_pm.D_u << "," << h_pm.D_v << "," << h_pm.a_u << "," << h_pm.b_v << "," << h_pm.type_switch << "," << h_pm.death_switch << "," << h_pm.u_death <<"\n";}

