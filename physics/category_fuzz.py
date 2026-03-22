import re
from rapidfuzz import process, fuzz

class ChemistryCategoryUtils:
    primary_categories = [
        "Inorganic Chemistry",
        "Organic Chemistry",
        "Analytical Chemistry",
        "Physical Chemistry and Chemical Physics",
        "Polymer Chemistry and Physics",
        "Nuclear Chemistry",
        "Applied and Interdisciplinary Chemistry"
    ]
    
    secondary_categories = {
        "Inorganic Chemistry": [
            "Elemental and Isotope Chemistry",
            "Coordination Chemistry",
            "Inorganic Synthesis and Separation Chemistry",
            "Solid-State and Physical Inorganic Chemistry",
            "Bioinorganic Chemistry"
        ],
        "Organic Chemistry": [
            "Organoelemental and Organometallic Chemistry",
            "Natural Products Chemistry",
            "Organic Synthesis",
            "Physical Organic and Organic Photochemistry",
            "Bioorganic Chemistry"
        ],
        "Analytical Chemistry": [
            "Chemical and Electrochemical Analysis",
            "Spectroscopy, Spectrometry, and Mass Spectrometry",
            "Chromatography and Separation Analysis",
            "Thermal, Radiochemical, and Phase Analysis",
            "Chemometrics"
        ],
        "Physical Chemistry and Chemical Physics": [
            "Chemical Thermodynamics and Thermochemistry",
            "Chemical Kinetics and Catalysis",
            "Quantum Chemistry, Structural Chemistry, and Computational Chemistry",
            "Colloid and Interface Chemistry",
            "Electrochemistry, Magnetochemistry, and Photochemistry"
        ],
        "Polymer Chemistry and Physics": [
            "Polymer Synthesis",
            "Physical Chemistry of Polymers (including Polymer Physics)",
            "Functional, Natural, and Inorganic Polymers"
        ],
        "Nuclear Chemistry": [
            "Radiochemistry and Environmental Radiochemistry",
            "Nuclear Reaction Chemistry (Fission, Fusion, Transmutation)"
        ],
        "Applied and Interdisciplinary Chemistry": [
            "Applied Chemistry and Chemical Engineering",
            "Chemical Biology",
            "Materials Chemistry (Nanochemistry, Carbon Chemistry, Soft Chemistry)",
            "History of Chemistry"
        ]
    }

    def normalize_text(self, s: str) -> str:
        s = s.lower()
        # 去数字、点、连字符、下划线、逗号、括号等，保留纯字母用于模糊匹配
        s = re.sub(r"[0-9\.\-\_\(\)\[\],&/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def fuzzy_match_label(self,
                          raw_label: str,
                          choices: list[str],
                          scorer=fuzz.WRatio,
                          threshold: int = 70) -> str | None:
        raw_clean = self.normalize_text(raw_label)
        # rapidfuzz extractOne 返回 (best_match, score, index)
        result = process.extractOne(
            query=raw_clean,
            choices=choices,
            scorer=scorer
        )
        if result:
            best, score, _ = result
            return best if score >= threshold else None
        return None

    def normalize_categories(
        self,
        raw_primary: str,
        raw_secondary: str,
        thresh_primary: int = 50,
        thresh_secondary: int = 50
    ) -> dict:
        """
        优先解析纯数字编号（如"1"、"1."或"1.1"），否则再走 fuzzy-match。
        返回 {'primary_category': ..., 'secondary_category': ...}。
        """
        primary_choices = self.primary_categories
        secondary_map = self.secondary_categories
        
        # 1) 尝试直接按 "X.Y" 解析子类编号
        m = re.match(r"^\s*(\d+)\s*\.\s*(\d+)\s*\.?\s*$", raw_secondary)
        if m:
            pi, si = int(m.group(1)), int(m.group(2))
            if 1 <= pi <= len(primary_choices):
                primary = primary_choices[pi - 1]
                secs = secondary_map.get(primary, [])
                if 1 <= si <= len(secs):
                    return {
                        "primary_category": primary,
                        "secondary_category": secs[si - 1]
                    }
                    
        # 2) 再尝试按 "X" 解析主类编号
        m = re.match(r"^\s*(\d+)\s*\.?\s*$", raw_primary)
        if m:
            pi = int(m.group(1))
            if 1 <= pi <= len(primary_choices):
                primary = primary_choices[pi - 1]
            else:
                primary = None
        else:
            primary = None

        # 3) 如果编号解析不到，再 fuzzy-match 主类
        if primary is None:
            primary = self.fuzzy_match_label(raw_primary, primary_choices, threshold=thresh_primary)

        # 如果主类依然没匹配上，直接返回空字符串
        if not primary:
            return {"primary_category": "", "secondary_category": ""}

        # 4) 在该主类对应的二级列表中做 fuzzy-match
        sec_choices = secondary_map.get(primary, [])
        secondary = self.fuzzy_match_label(raw_secondary, sec_choices, threshold=thresh_secondary)
        if not secondary:
            secondary = ""

        return {
            "primary_category": primary,
            "secondary_category": secondary
        }

    def category_hasher(self, primary: str, secondary: str) -> float:
        # 化学大类共 7 个，二级类最多 5 个，所以乘 8 空间足够且不冲突
        try:
            k = self.primary_categories.index(primary)
            m = self.secondary_categories[primary].index(secondary)
            return k * 8 + m
        except ValueError:
            return 170  # 匹配失败时的默认异常值

    def category_hasher_reverse(self, hash_val: float) -> tuple[str, str] | tuple[None, None]:
        if hash_val < 0 or hash_val > 56: # 化学最大索引为 6 * 8 + 4 = 52，56是个安全的边界
            return None, None
        k = int(hash_val // 8)
        m = int(hash_val % 8)
        try:
            primary = self.primary_categories[k]
            secondary = self.secondary_categories[primary][m]
            return primary, secondary
        except (IndexError, KeyError):
            return None, None
