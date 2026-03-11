#pragma once

#include <cctype>
#include <string>
#include <vector>

namespace mfftorch {

// Minimal periodic table mapping: symbol -> atomic number Z.
// We embed a full symbol table (1..118) so user can pass standard symbols
// in `pair_coeff * * core.pt H O ...`.
inline int symbol_to_Z(std::string sym) {
  // normalize: trim + capitalize first letter, lowercase rest
  auto trim = [](std::string& s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  };
  trim(sym);
  if (sym.empty()) return 0;
  sym[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(sym[0])));
  for (size_t i = 1; i < sym.size(); i++) sym[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(sym[i])));

  static const std::vector<const char*> Z_to_sym = {
      "X",
      "H","He",
      "Li","Be","B","C","N","O","F","Ne",
      "Na","Mg","Al","Si","P","S","Cl","Ar",
      "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
      "Ga","Ge","As","Se","Br","Kr",
      "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
      "In","Sn","Sb","Te","I","Xe",
      "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
      "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
      "Tl","Pb","Bi","Po","At","Rn",
      "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
      "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
      "Nh","Fl","Mc","Lv","Ts","Og",
  };

  for (int Z = 1; Z < static_cast<int>(Z_to_sym.size()); Z++) {
    if (sym == Z_to_sym[Z]) return Z;
  }
  return 0;
}

}  // namespace mfftorch

