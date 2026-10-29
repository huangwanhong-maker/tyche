// Checks of tyche/probability against reference values and closed forms.
//
// References. The values with many digits were computed with mpmath 1.3 at 40 significant digits
// (arbitrary-precision incomplete gamma and beta functions and bisection; an independent
// computation, 2026-10-24); each agrees with the classical tables at their printed precision:
// Abramowitz and Stegun (1964), Table 26.1 (Φ to 15 decimals) and Table 26.5; Fisher and Yates
// (1963), Statistical Tables, Tables III, IV and V (t, χ², F critical values to 3 or 4 figures);
// Wichura (1988), the test values printed with AS 241 (z for p = 0.25, 0.001, 10⁻²⁰). The
// rest are closed forms: the Cauchy and t₂ quantiles, χ²₂ as an exponential, F(1, ν) = t²_ν,
// beta and binomial sums with integer parameters, Poisson sums, and round trips F(F⁻¹(p)) = p.

#include <tyche/probability/distributions.hpp>
#include <tyche/probability/p_values.hpp>

#include <calx/verification/check.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>

using namespace tyche::probability;
using calx::verification::relative;

int main() {
  calx::verification::Suite s;

  // ---- normal ----
  const Normal z;
  s.check("normal: Φ(1) = 0.841344746068542949 (A&S Table 26.1)", relative(z.cdf(1), 0.84134474606854294859), 2e-16);
  s.check("normal: Φ(2) = 0.977249868051820793", relative(z.cdf(2), 0.9772498680518207928), 2e-16);
  s.check("normal: Φ(3) = 0.998650101968369905", relative(z.cdf(3), 0.99865010196836990547), 2e-16);
  s.check("normal: Φ(−10) = 7.6198530241605261e-24, relative, from the tail", relative(z.cdf(-10), 7.619853024160526066e-24), 1e-14);
  s.check("normal: P(Z > 30) = 4.906713927148187e-198, relative", relative(z.survival(30), 4.9067139271481870595e-198), 3e-13);  // z² ε from rounding z/√2
  s.check("normal: AS 241 at p = 0.25: −0.6744897501960817 (Wichura's test value)", z.quantile(0.25) + 0.6744897501960817432, 2e-16);
  s.check("normal: AS 241 at p = 0.001: −3.090232306167814", relative(z.quantile(0.001), -3.0902323061678135415), 2e-16);
  s.check("normal: AS 241 at p = 1e-20: −9.262340089798408", relative(z.quantile(1e-20), -9.2623400897984075737), 2e-16);
  s.check("normal: z₀.₉₇₅ = 1.959963984540054 (at the double nearest 0.975: …0538556)", relative(z.quantile(0.975), 1.959963984540053855604), 2e-16);
  s.check("normal: z₀.₉₉₉₉ = 3.719016485455681 (at the double nearest 0.9999: …5708387)", relative(z.quantile(0.9999), 3.719016485455708386723), 2e-16);
  s.check("normal: the upper quantile at q = 10⁻⁴, from the tail", relative(z.upper_quantile(1e-4), 3.7190164854556805644), 2e-16);
  s.check("normal: the upper quantile at q = 1e-300 is −Φ⁻¹(1e-300), kept from the tail", relative(z.upper_quantile(1e-300), 37.04709629936119924), 1e-14);
  {
    double worst = 0;
    // The relative error of Φ at z is about z² times the relative error of z (φ(z)/Φ(z) ≈ |z|).
    for (double p = 1e-12; p < 1; p *= 1.7) {
      const double q = z.quantile(p);
      worst = std::max(worst, relative(z.cdf(q), p) / (1 + q * q));
    }
    s.check("normal: Φ(Φ⁻¹(p)) = p from 1e-12 to 1, relative over 1 + z² (a slip in a coefficient of AS 241 would show here)", worst, 1e-15);
  }
  {
    const Normal n{10, 2};
    s.check("normal: N(10, 4)'s density at 11 is φ(½)/2", n.pdf(11) - std::exp(-0.125) / (2 * std::sqrt(2 * std::numbers::pi)), 1e-16);
    s.check("normal: and its 97.5% point is 10 + 2 z₀.₉₇₅", n.quantile(0.975) - (10 + 2 * 1.9599639845400542355), 4e-15);
  }

  // ---- Student's t ----
  s.check("t: t₀.₉₇₅,₁₀ = 2.228138851986275 (Fisher and Yates: 2.228)", relative(StudentT{10}.quantile(0.975), 2.2281388519862747484), 1e-13);
  s.check("t: t₀.₉₉₅,₅ = 4.032142983555228 (4.032)", relative(StudentT{5}.quantile(0.995), 4.0321429835552280784), 1e-13);
  s.check("t: t₀.₉₅,₁ = tan(0.45π) = 6.313751514675043, the Cauchy", relative(StudentT{1}.quantile(0.95), 6.313751514675043099), 1e-14);
  s.check("t: t₀.₉₇₅,₃₀ = 2.042272456301238 (2.042)", relative(StudentT{30}.quantile(0.975), 2.04227245630123831), 1e-13);
  s.check("t: t₀.₉₇₅ at ν = 3.5, fractional", relative(StudentT{3.5}.quantile(0.975), 2.9400886379827297938), 1e-13);
  s.check("t: t₀.₉₉₉₅,₄ = 8.610301581379275 (8.610)", relative(StudentT{4}.quantile(0.9995), 8.6103015813792750967), 1e-13);
  s.check("t: t₂'s quantile in closed form, (2p − 1)/√(2p(1 − p)) at p = 0.9", StudentT{2}.quantile(0.9) - 0.8 / std::sqrt(0.18), 1e-15);
  s.check("t: F(2.5) at ν = 7", relative(StudentT{7}.cdf(2.5), 0.97950389070712355156), 1e-14);
  s.check("t: P(T > 40) at ν = 3, relative", relative(StudentT{3}.survival(40), 1.7190340394579264142e-5), 1e-13);
  s.check("t: F(0.1) at ν = 50, near the centre", relative(StudentT{50}.cdf(0.1), 0.53962789317351496713), 1e-14);
  s.check("t: the density at 1.3, ν = 6", relative(StudentT{6}.pdf(1.3), 0.16057681451954479464), 1e-14);
  s.check("t: symmetry, F(−t) = 1 − F(t)", StudentT{4.5}.cdf(-1.7) - StudentT{4.5}.survival(1.7), 1e-16);
  s.check("t: ν = ∞ is the normal", StudentT{infinity}.quantile(0.975) - 1.959963984540053855604, 4e-16);

  // ---- χ² and gamma ----
  s.check("χ²: χ²₀.₉₅,₁ = 3.841458820694126 = z₀.₉₇₅² (Fisher and Yates: 3.841)", relative(ChiSquared{1}.quantile(0.95), 3.8414588206941259584), 1e-13);
  s.check("χ²: χ²₀.₉₅,₁₀ = 18.30703805327515 (18.307)", relative(ChiSquared{10}.quantile(0.95), 18.307038053275146872), 1e-13);
  s.check("χ²: χ²₀.₉₉,₃ = 11.34486673014437 (11.345)", relative(ChiSquared{3}.quantile(0.99), 11.344866730144371931), 1e-13);
  s.check("χ²: χ²₀.₀₅,₂₀ = 10.85081139418259 (10.851)", relative(ChiSquared{20}.quantile(0.05), 10.850811394182585163), 1e-13);
  s.check("χ²: the median at k = 0.7, fractional", relative(ChiSquared{0.7}.quantile(0.5), 0.21473897172368650474), 1e-12);
  s.check("χ²: P(χ²₁₀ > 100) = 5.44970198e-17, relative", relative(ChiSquared{10}.survival(100), 5.4497019829205293351e-17), 1e-13);
  s.check("χ²: F(3) at k = 4", relative(ChiSquared{4}.cdf(3), 0.44217459962892542767), 1e-14);
  s.check("χ²: the upper quantile at q = 1e-20, k = 5, from the tail", relative(ChiSquared{5}.upper_quantile(1e-20), 103.42897723775794862), 1e-12);
  s.check("χ²: k = 2 is the exponential of mean 2, F(x) = 1 − e^{−x/2}", ChiSquared{2}.cdf(3.3) + std::expm1(-1.65), 1e-15);
  s.check("χ²: and its quantile −2 log(1 − p)", relative(ChiSquared{2}.quantile(0.77), -2 * std::log1p(-0.77)), 1e-13);
  s.check("gamma: F(2.5) with shape 3, scale 0.5", relative(Gamma{3, 0.5}.cdf(2.5), 0.87534798051691885871), 1e-14);
  s.check("gamma: the 30% point with shape 2.5, scale 2", relative(Gamma{2.5, 2}.quantile(0.3), 2.9999081327599062821), 1e-12);
  s.check("gamma: the density with shape 1 is the exponential", Gamma{1, 2}.pdf(3) - 0.5 * std::exp(-1.5), 1e-16);
  s.check("gamma: mean and variance aθ, aθ²", std::fabs(Gamma{3, 2}.mean() - 6) + std::fabs(Gamma{3, 2}.variance() - 12), 0);

  // ---- F ----
  s.check("F: F₀.₉₅(3, 10) = 3.708264819046844 (Fisher and Yates: 3.71)", relative(FisherF{3, 10}.quantile(0.95), 3.7082648190468444854), 1e-12);
  s.check("F: F₀.₉₉(5, 20) = 4.102684630584733 (4.10)", relative(FisherF{5, 20}.quantile(0.99), 4.1026846305847330648), 1e-12);
  s.check("F: F₀.₉₅(1, 10) = t²₀.₉₇₅,₁₀ = 4.964602743730714", relative(FisherF{1, 10}.quantile(0.95), 4.9646027437307143713), 1e-12);
  s.check("F: F(1, ν) = t²: P(F > t²) = P(|T| > t)", FisherF{1, 7}.survival(2.5 * 2.5) - two_sided_t(2.5, 7), 1e-15);
  s.check("F: the lower 10% point of F(4, 7)", relative(FisherF{4, 7}.quantile(0.1), 0.25132155910077278104), 1e-12);
  s.check("F: P(F(4, 30) > 10), relative", relative(FisherF{4, 30}.survival(10), 2.8928437923539478172e-5), 1e-12);

  // ---- beta ----
  s.check("beta: I₀.₃(2, 5) = P(Bin(6, 0.3) ≥ 2) = 0.579825", Beta{2, 5}.cdf(0.3) - 0.579825, 1e-14);
  s.check("beta: the 90% point of Beta(2.5, 0.7)", relative(Beta{2.5, 0.7}.quantile(0.9), 0.98604772602954570028), 1e-12);
  s.check("beta: P(X > 0.999) for Beta(3, 4) = 1.497601e-11, from the tail", relative(Beta{3, 4}.survival(0.999), 1.497601e-11), 1e-11);
  s.check("beta: Beta(a, 1) has F(x) = xᵃ, quantile p^{1/a}", Beta{2.7, 1}.quantile(0.4) - std::pow(0.4, 1 / 2.7), 1e-14);

  // ---- Poisson and binomial ----
  s.check("poisson: P(X ≤ 3) at λ = 4.5, against the sum of the pmf", relative(Poisson{4.5}.cdf(3), 0.34229595583459106891), 1e-14);
  s.check("poisson: P(X > 30) at λ = 10, relative", relative(Poisson{10}.survival(30), 7.9837946599111854586e-8), 1e-12);
  {
    double sum = 0;
    for (std::int64_t k = 0; k <= 12; k++) sum += Poisson{6.2}.pmf(k);
    s.check("poisson: Σ pmf to 12 = F(12)", sum - Poisson{6.2}.cdf(12), 1e-15);
    const Poisson ps{6.2};
    const std::int64_t k = ps.quantile(0.9);
    s.require("poisson: the 90% point is the least k with F(k) ≥ 0.9", ps.cdf(k) >= 0.9 && ps.cdf(k - 1) < 0.9, std::to_string(k));
  }
  s.check("binomial: P(X ≤ 7) for n = 20, p = 0.3, against the sum of the pmf", relative(Binomial{20, 0.3}.cdf(7), 0.77227179741816046012), 1e-14);
  s.check("binomial: P(X > 15), relative", relative(Binomial{20, 0.3}.survival(15), 5.55025307829876e-6), 1e-12);
  s.check("binomial: pmf(5) for n = 10, p = ½ is 252/1024, relative", relative(Binomial{10, 0.5}.pmf(5), 252.0 / 1024), 1e-14);
  {
    const Binomial b{50, 0.37};
    const std::int64_t k = b.quantile(0.25);
    s.require("binomial: the quartile is the least k with F(k) ≥ ¼", b.cdf(k) >= 0.25 && b.cdf(k - 1) < 0.25, std::to_string(k));
  }

  // ---- round trips and p-values ----
  {
    double worst = 0;
    for (double p : {1e-10, 1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 0.9999}) {
      worst = std::max(worst, std::fabs(StudentT{3.3}.cdf(StudentT{3.3}.quantile(p)) - p) / std::min(p, 1 - p));
      worst = std::max(worst, std::fabs(FisherF{2.5, 9}.cdf(FisherF{2.5, 9}.quantile(p)) - p) / std::min(p, 1 - p));
      worst = std::max(worst, std::fabs(Beta{0.6, 3.2}.cdf(Beta{0.6, 3.2}.quantile(p)) - p) / std::min(p, 1 - p));
      worst = std::max(worst, std::fabs(Gamma{0.4, 3}.cdf(Gamma{0.4, 3}.quantile(p)) - p) / std::min(p, 1 - p));
    }
    s.check("quantiles: F(F⁻¹(p)) = p for t, F, beta and gamma, relative to min(p, 1 − p)", worst, 1e-11);
  }
  s.check("p-values: two-sided z at 1.96 is 0.04999579029644087", relative(two_sided_normal(1.96), 0.04999579029644086827), 1e-14);
  s.check("p-values: two-sided t is twice the upper tail", two_sided_t(-2.1, 12) - 2 * StudentT{12}.survival(2.1), 1e-16);
  s.check("p-values: the equal-tailed p of an asymmetric statistic, 2 min(F, S)", two_sided(ChiSquared{3}, 0.2) - 2 * ChiSquared{3}.cdf(0.2), 1e-16);
  s.require("p-values: capped at 1", two_sided(Normal{}, 0.0) == 1.0);
  bool threw = false;
  try { (void)Normal{0, -1}.cdf(0); } catch (const std::invalid_argument&) { threw = true; }
  s.require("parameters: σ ≤ 0 is refused", threw);
  return s.done();
}
