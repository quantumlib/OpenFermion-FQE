#include <complex.h>

void detect_cirq_sectors(double complex * cirq_wfn,
                         double thresh,
                         int * paramarray,
                         const int norb,
                         const int alpha_states,
                         const int beta_states,
                         const long long * cirq_aids,
                         const long long * cirq_bids,
                         const int * anumb,
                         const int * bnumb
                         );
