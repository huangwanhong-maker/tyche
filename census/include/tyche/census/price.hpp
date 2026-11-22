// The census of a population at two times with its genealogy, and the Price decomposition of the change of a character's mean
// (decision 0035).
//
// A census is a set of members, each with a weight x ≥ 0 (copies, capacity, employment, output, a budget share) and a character z (a
// productivity, a unit cost, a fitness, any measured property). A record joins a census before and a census after by a genealogy: each
// member after names its parents before with transmission shares φ_ji ≥ 0, and the part of it that descends from no one arrives, split by
// its sources, so that Σ_i φ_ji + Σ_sources a_j = 1. A member before with no descendant exits, with its reason; a member after is made by
// an operator (copying, imitation, mutation, crossover, continuing). Nothing here names what the members are: firms in an industry and
// candidate solutions in a search are both records of this form.
//
// With shares s_i = x_i / X and s'_j = x'_j / X', the mean Z = Σ s_i z_i, and p_j = Σ_i φ_ji the part of j that descends, each member
// before carries the share s'_i = Σ_j φ_ji s'_j of the census after and the character z'_i = Σ_j φ_ji s'_j z'_j / s'_i (z_i where
// s'_i = 0). The members with at least one descendant link are the continuing set C, those with none the exiters X; the parts that arrive
// are the entrants N. Then Z' − Z is decomposed exactly in one of four accounting forms (every one an identity, whatever the data):
//
//   survivors   (Price 1970, 1995; the reference of Melitz and Polanec 2015's dynamic Olley–Pakes decomposition, their eq. 7):
//               Z' − Z = (Z'_C − Z_C) + S'_N (Z'_N − Z'_C) + S_X (Z_C − Z_X),
//               the survivors' change split by Price's equation, W̄ (Z'_C − Z_C) = Cov_σ(w, z) + E_σ(w Δz), into selection
//               Σ_C (σ'_i − σ_i)(z_i − Z_C) and transmission Σ_C σ'_i Δz_i, with σ, σ' the shares normalised over C and w_i = x'_i / x_i;
//               selection split further over nested groups (the multilevel Price equation; Hodgson and Knudsen 2006, appendix);
//   incumbent   (Foster, Haltiwanger and Krizan 2001; Holm 2014, eq. 4): entrants and exiters judged against the mean before, R = Z:
//               Z' − Z = Σ_C Δs_i (z_i − R) + Σ_C s_i Δz_i + Σ_C Δs_i Δz_i + Σ_N s'_j (z'_j − R) − Σ_X s_i (z_i − R),
//               between (selection), within and cross (transmission), entry and exit;
//   average     (Griliches and Regev 1995): Z' − Z = Σ_C s̄_i Δz_i + Σ_C Δs_i (z̄_i − R) + Σ_N s'_j (z'_j − R) − Σ_X s_i (z_i − R)
//               with s̄, z̄ the averages over the two censuses and R = (Z + Z') / 2;
//   declared    the incumbent form against a reference R a caller declares. Since Σ_C Δs_i + S'_N − S_X = 0, any R is exact.
//
// The reference carries meaning: in a population whose every member's character grows by γ, the survivors' form credits entry and exit
// with nothing, the average form with s_N γ/2 and s_X γ/2, the incumbent form with s_N γ and nothing (Melitz and Polanec 2015, §2). The
// terms are an accounting, exact as identities and never causal: selection is a causal claim only when what makes the covariance is
// supplied (Hodgson and Knudsen 2006; Frank 2012). Transmission is credited to the operator that made each member after, entry to each
// source, exit to each reason. Records compose (two steps into one span); the selection terms of the steps do not add to the span's
// (Holm 2014, §4), which the checks show. The number type is a template: double, or morphe's exact rationals.
//
// Sources: G. R. Price, "Selection and covariance", Nature 227 (1970) 520–521; G. R. Price, "The nature of selection", J. Theor. Biol. 175
// (1995) 389–396; S. A. Frank, "Natural selection. IV. The Price equation", J. Evol. Biol. 25 (2012) 1002–1019; G. M. Hodgson and T.
// Knudsen, "The nature and units of social selection", J. Evol. Econ. 16 (2006) 477–489; J. R. Holm, "The significance of selection and
// learning for productivity growth", J. Evol. Econ. 24 (2014) 1013–1040 (doi 10.1007/s00191-014-0369-y); M. J. Melitz and S. Polanec,
// "Dynamic Olley–Pakes productivity decomposition with entry and exit", RAND J. Econ. 46 (2015) 362–375 (NBER WP 18182); L. Foster, J.
// Haltiwanger and C. J. Krizan, "Aggregate productivity growth", in New Developments in Productivity Analysis, Chicago 2001, 303–372; Z.
// Griliches and H. Regev, "Firm productivity in Israeli industry 1979–1988", J. Econometrics 65 (1995) 175–203.

#pragma once

#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyche::census {

namespace detail {
inline void require(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("tyche::census: " + what);
}
// Whether two sums agree: exactly for an exact number type, to a few units of rounding for floating point.
template <class T>
bool same(const T& a, const T& b, std::size_t n = 1) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(a - b) <= 1e-12 * static_cast<double>(n + 1) * (1 + std::fabs(a) + std::fabs(b));
  } else {
    return a == b;
  }
}
}  // namespace detail

// A census: each member's weight and character, and optionally its group at each declared level (outermost first).
template <class T>
struct Census {
  std::vector<T> weight, trait;
  std::vector<std::vector<std::string>> levels;  // empty, or one list of labels per member, every list of the same length
  std::size_t size() const { return weight.size(); }
};

template <class T>
struct Link {
  std::size_t parent;  // index in the census before
  T share;             // φ_ji
};
template <class T>
struct Arrival {
  std::string source;
  T share;  // the part of the member that arrives from this source
};

template <class T>
struct Record {
  Census<T> before, after;
  std::vector<std::vector<Link<T>>> parents;      // per member after
  std::vector<std::vector<Arrival<T>>> arrivals;  // per member after; shares and links sum to one
  std::vector<std::string> made_by;               // per member after: the operator that made its descended part
  std::vector<std::string> exit_reason;           // per member before (read where it has no descendant)
};

enum class Form { survivors, incumbent, average, declared };

inline const char* name(Form f) {
  switch (f) {
    case Form::survivors: return "survivors";
    case Form::incumbent: return "incumbent";
    case Form::average: return "average";
    case Form::declared: return "declared";
  }
  return "";
}

template <class T>
struct Decomposition {
  Form form = Form::survivors;
  T before_mean{}, after_mean{}, total{};  // Z, Z', Z' − Z
  T reference{};                           // R of the incumbent, average and declared forms
  T selection{}, transmission{}, entry{}, exit{};
  T within{}, cross{};                     // the incumbent and declared forms' split of transmission (within + cross); the average form's
                                           // transmission is its within term, and cross is nought
  // The survivors' form: Price's equation over the continuing set, W̄ (Z'_C − Z_C) = Cov + E, and selection by level (between groups at
  // each declared level, outermost first, then within the finest groups), which sums to selection.
  T continuing_change{}, mean_fitness{}, covariance{}, expectation{};
  std::vector<T> selection_by_level;
  std::map<std::string, T> transmission_by_operator, entry_by_source, exit_by_reason;
  T entering_share{}, exiting_share{};     // S'_N, S_X
};

// Checks a record and refuses what is not one, naming the fault.
template <class T>
void validate(const Record<T>& r) {
  const std::size_t n = r.before.size(), m = r.after.size();
  detail::require(r.before.trait.size() == n && r.after.trait.size() == m, "every member has one weight and one character");
  detail::require(r.parents.size() == m && r.arrivals.size() == m && r.made_by.size() == m, "parents, arrivals and operators for every member after");
  detail::require(r.exit_reason.empty() || r.exit_reason.size() == n, "an exit reason for every member before, or none");
  detail::require(n > 0 && m > 0, "two censuses with members");
  T X{}, Xp{};
  for (const auto& x : r.before.weight) detail::require(!(x < T(0)), "weights are not negative"), X += x;
  for (const auto& x : r.after.weight) detail::require(!(x < T(0)), "weights are not negative"), Xp += x;
  detail::require(T(0) < X && T(0) < Xp, "each census has a positive total weight");
  if (!r.before.levels.empty()) {
    detail::require(r.before.levels.size() == n, "levels for every member before, or none");
    for (const auto& l : r.before.levels) detail::require(l.size() == r.before.levels[0].size(), "every member's levels have one length");
  }
  for (std::size_t j = 0; j < m; j++) {
    T sum{};
    for (const auto& L : r.parents[j]) {
      detail::require(L.parent < n, "a parent is a member of the census before");
      detail::require(!(L.share < T(0)), "transmission shares are not negative");
      sum += L.share;
    }
    for (const auto& a : r.arrivals[j]) detail::require(!(a.share < T(0)), "arriving shares are not negative"), sum += a.share;
    detail::require(detail::same(sum, T(1), r.parents[j].size() + r.arrivals[j].size()), "a member's links and arrivals sum to one");
  }
  for (std::size_t j = 0; j < m; j++)
    for (const auto& L : r.parents[j])
      detail::require(T(0) < r.before.weight[L.parent] || !(T(0) < L.share), "an ancestor of a member after has a positive weight");
}

namespace detail {
template <class T>
struct Prepared {
  std::vector<T> s, sp;          // shares before, after
  std::vector<T> sa, za;         // per member before: s'_i and z'_i (z_i where s'_i = 0)
  std::vector<bool> continuing;  // per member before
  std::vector<T> p;              // per member after: the part that descends
  T Z{}, Zp{};
};

template <class T>
Prepared<T> prepare(const Record<T>& r) {
  validate(r);
  const std::size_t n = r.before.size(), m = r.after.size();
  Prepared<T> P;
  T X{}, Xp{};
  for (const auto& x : r.before.weight) X += x;
  for (const auto& x : r.after.weight) Xp += x;
  P.s.resize(n), P.sp.resize(m), P.sa.assign(n, T(0)), P.za.assign(n, T(0)), P.continuing.assign(n, false), P.p.assign(m, T(0));
  for (std::size_t i = 0; i < n; i++) P.s[i] = r.before.weight[i] / X, P.Z += P.s[i] * r.before.trait[i];
  for (std::size_t j = 0; j < m; j++) P.sp[j] = r.after.weight[j] / Xp, P.Zp += P.sp[j] * r.after.trait[j];
  std::vector<T> szp(n, T(0));
  for (std::size_t j = 0; j < m; j++)
    for (const auto& L : r.parents[j]) {
      if (!(T(0) < L.share)) continue;
      P.continuing[L.parent] = true;
      P.sa[L.parent] += L.share * P.sp[j];
      szp[L.parent] += L.share * P.sp[j] * r.after.trait[j];
      P.p[j] += L.share;
    }
  for (std::size_t i = 0; i < n; i++) P.za[i] = T(0) < P.sa[i] ? szp[i] / P.sa[i] : r.before.trait[i];
  return P;
}

// Between-group selection at each level and within the finest groups, over the members `idx` of one group at depth `depth`, with
// shares b (before) and a (after) normalised over the continuing set and weight `outer` = the group's normalised share after.
template <class T>
void levels(const Record<T>& r, const std::vector<std::size_t>& idx, std::size_t depth, const std::vector<T>& b, const std::vector<T>& a,
            const T& outer, std::vector<T>& out) {
  T B{}, A{}, Zs{};
  for (auto i : idx) B += b[i], A += a[i], Zs += b[i] * r.before.trait[i];
  Zs = Zs / B;
  const std::size_t L = r.before.levels.empty() ? 0 : r.before.levels[0].size();
  if (depth == L) {
    T sel{};
    for (auto i : idx) sel += (a[i] / A - b[i] / B) * (r.before.trait[i] - Zs);
    out[depth] += outer * sel;
    return;
  }
  std::map<std::string, std::vector<std::size_t>> groups;
  for (auto i : idx) groups[r.before.levels[i][depth]].push_back(i);
  for (const auto& [label, g] : groups) {
    T Bk{}, Ak{}, Zk{};
    for (auto i : g) Bk += b[i], Ak += a[i], Zk += b[i] * r.before.trait[i];
    Zk = Zk / Bk;
    out[depth] += outer * (Ak / A - Bk / B) * (Zk - Zs);
    if (T(0) < Ak) levels(r, g, depth + 1, b, a, outer * Ak / A, out);
  }
}
}  // namespace detail

// The decomposition of Z' − Z in the form asked; `reference` is read by the declared form alone.
template <class T>
Decomposition<T> decompose(const Record<T>& r, Form form = Form::survivors, const T& reference = T(0)) {
  const auto P = detail::prepare(r);
  const std::size_t n = r.before.size(), m = r.after.size();
  Decomposition<T> D;
  D.form = form, D.before_mean = P.Z, D.after_mean = P.Zp, D.total = P.Zp - P.Z;
  T SC{}, SpC{}, ZC{}, ZpC{};
  for (std::size_t i = 0; i < n; i++)
    if (P.continuing[i]) SC += P.s[i], ZC += P.s[i] * r.before.trait[i], SpC += P.sa[i], ZpC += P.sa[i] * P.za[i];
  for (std::size_t i = 0; i < n; i++)
    if (!P.continuing[i]) D.exiting_share += P.s[i];
  for (std::size_t j = 0; j < m; j++) D.entering_share += (T(1) - P.p[j]) * P.sp[j];
  auto reason = [&](std::size_t i) { return r.exit_reason.empty() || r.exit_reason[i].empty() ? std::string("exit") : r.exit_reason[i]; };

  if (form == Form::survivors) {
    detail::require(T(0) < SC && T(0) < SpC, "the survivors' form needs members that continue, with weight before and after");
    ZC = ZC / SC, ZpC = ZpC / SpC;
    D.continuing_change = ZpC - ZC;
    // Selection and transmission over C, the shares normalised over C.
    std::vector<T> b(n, T(0)), a(n, T(0));
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < n; i++)
      if (P.continuing[i]) b[i] = P.s[i] / SC, a[i] = P.sa[i] / SpC, idx.push_back(i), D.selection += (a[i] - b[i]) * (r.before.trait[i] - ZC);
    for (std::size_t j = 0; j < m; j++)
      for (const auto& L : r.parents[j]) {
        if (!(T(0) < L.share)) continue;
        const T c = L.share * P.sp[j] * (r.after.trait[j] - r.before.trait[L.parent]) / SpC;
        D.transmission += c, D.transmission_by_operator[r.made_by[j]] += c;
      }
    // Price's equation in its fitness form: w_i = x'_i / x_i with x'_i = Σ_j φ_ji x'_j over the continuing set.
    T XC{}, XpC{};
    for (std::size_t i = 0; i < n; i++)
      if (P.continuing[i]) XC += r.before.weight[i];
    std::vector<T> xa(n, T(0));
    for (std::size_t j = 0; j < m; j++)
      for (const auto& L : r.parents[j]) xa[L.parent] += L.share * r.after.weight[j];
    for (std::size_t i = 0; i < n; i++)
      if (P.continuing[i]) XpC += xa[i];
    D.mean_fitness = XpC / XC;
    for (std::size_t i = 0; i < n; i++)
      if (P.continuing[i]) {
        const T w = xa[i] / r.before.weight[i];
        D.covariance += b[i] * (w - D.mean_fitness) * (r.before.trait[i] - ZC);
        D.expectation += b[i] * w * (P.za[i] - r.before.trait[i]);
      }
    // Selection by level.
    const std::size_t L = r.before.levels.empty() ? 0 : r.before.levels[0].size();
    D.selection_by_level.assign(L + 1, T(0));
    detail::levels(r, idx, 0, b, a, T(1), D.selection_by_level);
    // Entry against the survivors after, exit against the survivors before.
    for (std::size_t j = 0; j < m; j++)
      for (const auto& A : r.arrivals[j]) {
        const T c = A.share * P.sp[j] * (r.after.trait[j] - ZpC);
        D.entry += c, D.entry_by_source[A.source] += c;
      }
    for (std::size_t i = 0; i < n; i++)
      if (!P.continuing[i]) {
        const T c = P.s[i] * (ZC - r.before.trait[i]);
        D.exit += c, D.exit_by_reason[reason(i)] += c;
      }
    return D;
  }

  const T R = form == Form::incumbent ? P.Z : form == Form::average ? (P.Z + P.Zp) / T(2) : reference;
  D.reference = R;
  for (std::size_t i = 0; i < n; i++) {
    if (!P.continuing[i]) continue;
    const T ds = P.sa[i] - P.s[i], dz = P.za[i] - r.before.trait[i];
    if (form == Form::average) {
      D.selection += ds * ((r.before.trait[i] + P.za[i]) / T(2) - R);
      D.within += (P.s[i] + P.sa[i]) / T(2) * dz;
    } else {
      D.selection += ds * (r.before.trait[i] - R);
      D.within += P.s[i] * dz, D.cross += ds * dz;
    }
  }
  D.transmission = D.within + D.cross;
  // Transmission credited to operators, pair by pair: the incumbent and declared forms' within + cross is Σ_C s'_i Δz_i, the pair (j, i)
  // contributing φ_ji s'_j (z'_j − z_i); the average form's Σ_C s̄_i Δz_i weights each pair by (1 + s_i / s'_i) / 2.
  for (std::size_t j = 0; j < m; j++)
    for (const auto& L : r.parents[j]) {
      if (!(T(0) < L.share) || !(T(0) < P.sa[L.parent])) continue;
      T c = L.share * P.sp[j] * (r.after.trait[j] - r.before.trait[L.parent]);
      if (form == Form::average) c = c * (T(1) + P.s[L.parent] / P.sa[L.parent]) / T(2);
      D.transmission_by_operator[r.made_by[j]] += c;
    }
  for (std::size_t j = 0; j < m; j++)
    for (const auto& A : r.arrivals[j]) {
      const T c = A.share * P.sp[j] * (r.after.trait[j] - R);
      D.entry += c, D.entry_by_source[A.source] += c;
    }
  for (std::size_t i = 0; i < n; i++)
    if (!P.continuing[i]) {
      const T c = -P.s[i] * (r.before.trait[i] - R);
      D.exit += c, D.exit_by_reason[reason(i)] += c;
    }
  return D;
}

// The change of the character's variance over the census: Var' − Var = Δ(mean of z²) − (Z'² − Z²), the first decomposed as any mean is.
template <class T>
struct Moment {
  T before{}, after{}, change{};  // the variances (share-weighted) and their change
  Decomposition<T> of_square;     // the decomposition of the mean of z²
  T mean_shift{};                 // Z'² − Z²
};

template <class T>
Moment<T> second_moment(const Record<T>& r, Form form = Form::survivors, const T& reference = T(0)) {
  Record<T> q = r;
  for (auto& z : q.before.trait) z = z * z;
  for (auto& z : q.after.trait) z = z * z;
  Moment<T> M;
  M.of_square = decompose(q, form, reference);
  const auto D = decompose(r, Form::incumbent);
  M.mean_shift = D.after_mean * D.after_mean - D.before_mean * D.before_mean;
  M.before = M.of_square.before_mean - D.before_mean * D.before_mean;
  M.after = M.of_square.after_mean - D.after_mean * D.after_mean;
  M.change = M.after - M.before;
  return M;
}

// Two steps joined into one span, the census after the first being the census before the second: transmission shares multiply along
// lineages, the parts that arrived in either step arrive in the span with their sources, and a member that left no descendant at the end
// exits with its own reason or those of its descendants. Transmission over a span is credited to "composed".
template <class T>
Record<T> compose(const Record<T>& r1, const Record<T>& r2) {
  validate(r1), validate(r2);
  detail::require(r1.after.size() == r2.before.size(), "the second step begins from the census the first ends with");
  for (std::size_t k = 0; k < r1.after.size(); k++)
    detail::require(r1.after.weight[k] == r2.before.weight[k] && r1.after.trait[k] == r2.before.trait[k], "the middle census is the same in both steps");
  Record<T> c;
  c.before = r1.before, c.after = r2.after;
  const std::size_t n = r1.before.size(), m = r2.after.size();
  c.parents.assign(m, {}), c.arrivals.assign(m, {}), c.made_by.assign(m, "composed");
  for (std::size_t k = 0; k < m; k++) {
    std::map<std::size_t, T> phi;
    std::map<std::string, T> come;
    for (const auto& a : r2.arrivals[k]) come[a.source] += a.share;
    for (const auto& L2 : r2.parents[k]) {
      for (const auto& L1 : r1.parents[L2.parent]) phi[L1.parent] += L2.share * L1.share;
      for (const auto& a : r1.arrivals[L2.parent]) come[a.source] += L2.share * a.share;
    }
    for (const auto& [i, s] : phi)
      if (T(0) < s) c.parents[k].push_back({i, s});
    for (const auto& [src, s] : come)
      if (T(0) < s) c.arrivals[k].push_back({src, s});
  }
  // Exit reasons in the span.
  std::vector<bool> survives(r2.before.size(), false);
  for (std::size_t k = 0; k < m; k++)
    for (const auto& L : r2.parents[k])
      if (T(0) < L.share) survives[L.parent] = true;
  c.exit_reason.assign(n, "exit");
  std::vector<std::set<std::string>> why(n);
  std::vector<bool> linked(n, false);
  for (std::size_t j = 0; j < r1.after.size(); j++)
    for (const auto& L : r1.parents[j])
      if (T(0) < L.share) {
        linked[L.parent] = true;
        if (!survives[j]) why[L.parent].insert(r2.exit_reason.empty() || r2.exit_reason[j].empty() ? std::string("exit") : r2.exit_reason[j]);
      }
  for (std::size_t i = 0; i < n; i++) {
    if (!linked[i]) {
      c.exit_reason[i] = r1.exit_reason.empty() || r1.exit_reason[i].empty() ? std::string("exit") : r1.exit_reason[i];
    } else if (!why[i].empty()) {
      std::string s;
      for (const auto& w : why[i]) s += (s.empty() ? "" : "; ") + w;
      c.exit_reason[i] = s;
    }
  }
  return c;
}

// A record from two panels keyed by identity: a member before and a member after with the same id continue (share one, made by
// `made_by`); members after with an id not seen before arrive from `entry_source`; members before not seen after exit for `exit_reason`.
template <class T>
Record<T> by_identity(const std::vector<std::string>& ids_before, const Census<T>& before, const std::vector<std::string>& ids_after,
                      const Census<T>& after, const std::string& entry_source = "entry", const std::string& exit_reason = "exit",
                      const std::string& made_by = "continuing") {
  detail::require(ids_before.size() == before.size() && ids_after.size() == after.size(), "an id for every member");
  std::map<std::string, std::size_t> at;
  for (std::size_t i = 0; i < ids_before.size(); i++) detail::require(at.emplace(ids_before[i], i).second, "ids before are distinct: " + ids_before[i]);
  Record<T> r;
  r.before = before, r.after = after;
  r.parents.assign(after.size(), {}), r.arrivals.assign(after.size(), {}), r.made_by.assign(after.size(), made_by);
  r.exit_reason.assign(before.size(), exit_reason);
  std::set<std::string> seen;
  for (std::size_t j = 0; j < ids_after.size(); j++) {
    detail::require(seen.insert(ids_after[j]).second, "ids after are distinct: " + ids_after[j]);
    const auto it = at.find(ids_after[j]);
    if (it != at.end()) r.parents[j].push_back({it->second, T(1)});
    else r.arrivals[j].push_back({entry_source, T(1)});
  }
  return r;
}

}  // namespace tyche::census
