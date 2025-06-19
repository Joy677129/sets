import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import re
import itertools
import openai
import json
import os

# --- Streamlit App: Set Operations, Power Set, and Advanced Set Definition Validator ---
st.set_page_config(page_title="Set Operations & Advanced Set Definition Validator", layout="wide")
st.title("🔁 Interactive Set Operations, Power Set Generator & Set Definition Validator")

# --- Helper Functions ---
@st.cache_data
def parse_element(x: str):
    """Parse a single element: strip whitespace, try to convert to int, else keep as string."""
    x_str = x.strip()
    if x_str == "":
        return None
    try:
        return int(x_str)
    except ValueError:
        return x_str

@st.cache_data
def parse_set(text: str):
    """Parse comma-separated elements into a Python set."""
    result = set()
    for part in text.split(','):
        el = parse_element(part)
        if el is not None:
            result.add(el)
    return result

@st.cache_data
def parse_universal(text: str):
    """Parse universal set input: numeric range 'start-end' or comma-separated list."""
    txt = text.strip()
    range_match = re.fullmatch(r"\s*([-]?\d+)\s*-\s*([-]?\d+)\s*", txt)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start <= end:
            return set(range(start, end+1))
        else:
            return set(range(end, start+1))
    return parse_set(txt)

@st.cache_data
def sort_result(s: set):
    """Sort a set for consistent display; fallback to string keys for mixed types."""
    try:
        return sorted(s)
    except TypeError:
        return sorted(s, key=lambda x: str(x))

def format_set(s: set) -> str:
    """Format a set as a math-like string: ∅ or {a, b, ...}."""
    if not s:
        return "∅"
    elems = sort_result(s)
    return "{" + ", ".join(str(e) for e in elems) + "}"

# Color blending for Venn diagrams
def blend_colors(color1, color2):
    import matplotlib.colors as mcolors
    rgb1 = mcolors.to_rgb(color1)
    rgb2 = mcolors.to_rgb(color2)
    return tuple((c1 + c2) / 2 for c1, c2 in zip(rgb1, rgb2))

def blend_colors_3(color1, color2, color3):
    import matplotlib.colors as mcolors
    rgb1 = mcolors.to_rgb(color1)
    rgb2 = mcolors.to_rgb(color2)
    rgb3 = mcolors.to_rgb(color3)
    return tuple((c1 + c2 + c3) / 3 for c1, c2, c3 in zip(rgb1, rgb2, rgb3))

# --- ChatGPT API Integration for Set Definition Parsing ---
# Ensure OPENAI_API_KEY is set in environment or Streamlit secrets
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not openai_api_key:
    st.warning("OPENAI_API_KEY not found. GPT-based set definition validation will not work.")
else:
    openai.api_key = openai_api_key

@st.cache_data(show_spinner=False)
def interpret_set_definition_via_gpt(user_input: str) -> dict:
    """Call ChatGPT API to parse arbitrary set definition, return dict with keys: valid, type, elements, description, note."""
    system_msg = "You are a helper that interprets mathematical set definitions given in arbitrary string form. Respond ONLY with valid JSON."
    examples = [
        {"input": "1, 2, 2, 3", "output": {"valid": True, "type": "roster", "elements": [1, 2, 3], "description": "Finite set with elements 1, 2, 3 (duplicates ignored)", "note": "Duplicates removed"}},
        {"input": "5-10", "output": {"valid": True, "type": "range", "elements": [5, 6, 7, 8, 9, 10], "description": "Integers from 5 to 10 inclusive", "note": ""}},
        {"input": "{ x | x is even and x < 10 }", "output": {"valid": True, "type": "predicate", "elements": [0, 2, 4, 6, 8], "description": "Even non-negative integers less than 10", "note": "Enumerated finite elements"}},
        {"input": "all primes", "output": {"valid": True, "type": "predicate", "elements": None, "description": "Set of all prime numbers", "note": "Infinite set; cannot enumerate fully"}},
        {"input": "{1, {2}}", "output": {"valid": False, "type": "invalid", "elements": None, "description": "", "note": "Nested sets not supported"}}
    ]
    messages = [{"role": "system", "content": system_msg}]
    for ex in examples:
        prompt_user = f"Input: \"{ex['input']}\"\nJSON:"
        messages.append({"role": "user", "content": prompt_user})
        messages.append({"role": "assistant", "content": json.dumps(ex['output'])})
    prompt_actual = f"Input: \"{user_input}\"\nJSON:"
    messages.append({"role": "user", "content": prompt_actual})
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)
        if not isinstance(data, dict) or 'valid' not in data:
            return {"valid": False, "type": "unknown", "elements": None, "description": "", "note": "Invalid response format"}
        elems = data.get("elements")
        if isinstance(elems, list):
            parsed = []
            for e in elems:
                try:
                    parsed.append(int(e))
                except Exception:
                    parsed.append(e)
            data["elements"] = parsed
        return data
    except Exception as e:
        # Handle insufficient quota or other OpenAI errors gracefully
        err_str = str(e)
        if 'insufficient_quota' in err_str or 'quota' in err_str.lower():
            note = "Insufficient quota or quota exceeded. GPT-based validation unavailable."
        else:
            note = f"Error: {e}"
        return {"valid": False, "type": "error", "elements": None, "description": "", "note": note}

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
show_venn = st.sidebar.checkbox("Enable Venn Diagram Operations", True)
show_powerset = st.sidebar.checkbox("Enable Power Set Generator", False)
show_validator = st.sidebar.checkbox("Enable Set Definition Validator", False)

# --- Venn Diagram Section ---
if show_venn:
    st.sidebar.subheader("Venn Diagram Settings")
    num_sets = st.sidebar.selectbox("Number of Sets for Venn Diagram", [2, 3], index=0)
    default_examples = {2: ["2, 4, 6, 8", "3, 4, 5, 6, 9"], 3: ["1, 2, 3, 5", "2, 3, 4, 6", "1, 4, 5, 7"]}
    A_input = st.sidebar.text_input("Set A elements (comma-separated)", value=default_examples[num_sets][0])
    B_input = st.sidebar.text_input("Set B elements (comma-separated)", value=default_examples[num_sets][1])
    C_input = None
    if num_sets == 3:
        C_input = st.sidebar.text_input("Set C elements (comma-separated)", value=default_examples[3][2])
    color_A = st.sidebar.color_picker("Color for Set A", "#FF5733")
    color_B = st.sidebar.color_picker("Color for Set B", "#33C1FF")
    color_C = None
    if num_sets == 3:
        color_C = st.sidebar.color_picker("Color for Set C", "#75FF33")
    U_input = st.sidebar.text_input("Universal Set (e.g., 1-10 or comma-separated)", value="1-10")
    if num_sets == 2:
        operation = st.sidebar.selectbox(
            "Choose Operation (2 sets)", [
                "A ∪ B", "A ∩ B", "A \\ B", "B \\ A",
                "Complement of A", "Complement of B",
                "Complement of (A ∪ B)", "Complement of (A ∩ B)"
            ]
        )
    else:
        operation = st.sidebar.selectbox(
            "Choose Operation (3 sets)", [
                "A ∪ B ∪ C", "A ∩ B ∩ C",
                "A \\ (B ∪ C)", "B \\ (A ∪ C)", "C \\ (A ∪ B)",
                "(A ∩ B) \\ C", "(A ∩ C) \\ B", "(B ∩ C) \\ A",
                "Complement of A", "Complement of B", "Complement of C",
                "Complement of (A ∪ B ∪ C)", "Complement of (A ∩ B ∩ C)"
            ]
        )
    A = parse_set(A_input)
    B = parse_set(B_input)
    C = parse_set(C_input) if num_sets == 3 else None
    U = parse_universal(U_input)
    if 'Complement' in operation:
        missing = set()
        missing |= (A - U); missing |= (B - U)
        if num_sets == 3: missing |= (C - U)
        if missing:
            st.sidebar.error(f"Universal set missing elements: {format_set(missing)}. Complements may be incomplete.")
    result, highlight_ids = set(), []
    if show_venn and num_sets == 2:
        set1, set2 = A, B; colors = (color_A, color_B)
        if operation == "A ∪ B": result = A | B; highlight_ids = ['10','01','11']
        elif operation == "A ∩ B": result = A & B; highlight_ids = ['11']
        elif operation == "A \\ B": result = A - B; highlight_ids = ['10']
        elif operation == "B \\ A": result = B - A; highlight_ids = ['01']
        elif operation == "Complement of A": result = U - A; set1, set2 = U, A; colors = ("#CCCCCC", color_A); highlight_ids = ['10']
        elif operation == "Complement of B": result = U - B; set1, set2 = U, B; colors = ("#CCCCCC", color_B); highlight_ids = ['10']
        elif operation == "Complement of (A ∪ B)": union = A | B; result = U - union; set1, set2 = U, union; colors=("#CCCCCC","#888888"); highlight_ids=['10']
        elif operation == "Complement of (A ∩ B)": inter = A & B; result = U - inter; set1, set2 = U, inter; colors=("#CCCCCC","#888888"); highlight_ids=['10']
        else: st.error("Unknown operation."); set1=set2=result=set(); colors=(color_A, color_B)
        result_repr = format_set(result)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Venn Diagram")
            def plot_venn2_local(s1, s2, c1, c2, hids, repr_str):
                fig, ax = plt.subplots()
                v = venn2([s1, s2], set_labels=("A", "B"), set_colors=(c1, c2), ax=ax)
                normal_lw, inter_lw, result_lw = 0.5, 1.0, 1.5
                for rid, face_color in [('10', c1), ('01', c2)]:
                    patch = v.get_patch_by_id(rid)
                    if patch:
                        patch.set_facecolor(face_color); patch.set_alpha(1.0)
                        lw = result_lw if rid in hids else normal_lw
                        patch.set_edgecolor('black'); patch.set_linewidth(lw)
                p11 = v.get_patch_by_id('11')
                if p11:
                    blended = blend_colors(c1, c2)
                    p11.set_facecolor(blended); p11.set_alpha(1.0)
                    lw = result_lw if '11' in hids else inter_lw
                    p11.set_edgecolor('black'); p11.set_linewidth(lw)
                ax.set_title(f"Result: {repr_str}", fontsize=10)
                st.pyplot(fig)
            plot_venn2_local(set1, set2, colors[0], colors[1], highlight_ids, result_repr)
        with col2:
            st.subheader("Resulting Set")
            if result:
                st.write(f"Result: {result_repr}")
                with st.expander("Detailed Regions"):
                    A_only = A - B; B_only = B - A; AB = A & B
                    regions = {'A only': (A_only, '10'), 'B only': (B_only, '01'), 'A ∩ B': (AB, '11')}
                    for label, (subset, rid) in regions.items():
                        incl = '✅' if rid in highlight_ids else '❌'
                        st.write(f"**{label}**: {format_set(subset)} {incl}")
            else:
                st.info("Resulting set is empty (∅).")
    if show_venn and num_sets == 3:
        set1, set2, set3 = A, B, C; colors = (color_A, color_B, color_C)
        if operation == "A ∪ B ∪ C": result = A | B | C; highlight_ids=['100','010','001','110','101','011','111']
        elif operation == "A ∩ B ∩ C": result = A & B & C; highlight_ids=['111']
        elif operation == "A \\ (B ∪ C)": result = A - (B | C); highlight_ids=['100']
        elif operation == "B \\ (A ∪ C)": result = B - (A | C); highlight_ids=['010']
        elif operation == "C \\ (A ∪ B)": result = C - (A | B); highlight_ids=['001']
        elif operation == "(A ∩ B) \\ C": result = (A & B) - C; highlight_ids=['110']
        elif operation == "(A ∩ C) \\ B": result = (A & C) - B; highlight_ids=['101']
        elif operation == "(B ∩ C) \\ A": result = (B & C) - A; highlight_ids=['011']
        elif operation == "Complement of A": result = U - A; set1, set2, set3 = U, A, set(); colors=("#CCCCCC",color_A,"#EEEEEE"); highlight_ids=['100']
        elif operation == "Complement of B": result = U - B; set1, set2, set3 = U, B, set(); colors=("#CCCCCC",color_B,"#EEEEEE"); highlight_ids=['100']
        elif operation == "Complement of C": result = U - C; set1, set2, set3 = U, C, set(); colors=("#CCCCCC",color_C,"#EEEEEE"); highlight_ids=['100']
        elif operation == "Complement of (A ∪ B ∪ C)": union=A|B|C; result=U-union; set1, set2, set3=U,union,set(); colors=("#CCCCCC","#888888","#EEEEEE"); highlight_ids=['100']
        elif operation == "Complement of (A ∩ B ∩ C)": inter=A&B&C; result=U-inter; set1, set2, set3=U,inter,set(); colors=("#CCCCCC","#888888","#EEEEEE"); highlight_ids=['100']
        else: st.error("Unknown operation."); set1=set2=set3=result=set(); colors=(color_A, color_B, color_C)
        result_repr = format_set(result)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Venn Diagram")
            def plot_venn3_local(s1, s2, s3, c1, c2, c3, hids, repr_str):
                fig, ax = plt.subplots()
                v = venn3([s1, s2, s3], set_labels=("A","B","C"), set_colors=(c1,c2,c3), ax=ax)
                normal_lw, inter2_lw, inter3_lw, result_lw = 0.5, 1.0, 1.2, 1.5
                region_colors={'100':c1,'010':c2,'001':c3,'110':blend_colors(c1,c2),'101':blend_colors(c1,c3),'011':blend_colors(c2,c3),'111':blend_colors_3(c1,c2,c3)}
                for rid, patch in [(rid, v.get_patch_by_id(rid)) for rid in region_colors]:
                    if patch:
                        patch.set_facecolor(region_colors[rid]); patch.set_alpha(1.0)
                        if rid=='111': lw=result_lw if rid in hids else inter3_lw
                        elif rid in ['110','101','011']: lw=result_lw if rid in hids else inter2_lw
                        else: lw=result_lw if rid in hids else normal_lw
                        patch.set_edgecolor('black'); patch.set_linewidth(lw)
                ax.set_title(f"Result: {repr_str}", fontsize=10)
                st.pyplot(fig)
            plot_venn3_local(set1, set2, set3, colors[0], colors[1], colors[2], highlight_ids, result_repr)
        with col2:
            st.subheader("Resulting Set")
            if result:
                st.write(f"Result: {result_repr}")
                with st.expander("Detailed Regions"):
                    onlyA=A-B-C; onlyB=B-A-C; onlyC=C-A-B; AB_only=(A&B)-C; AC_only=(A&C)-B; BC_only=(B&C)-A; ABC=A&B&C
                    regions={'A only':(onlyA,'100'),'B only':(onlyB,'010'),'C only':(onlyC,'001'),'A ∩ B only':(AB_only,'110'),'A ∩ C only':(AC_only,'101'),'B ∩ C only':(BC_only,'011'),'A ∩ B ∩ C':(ABC,'111')}
                    for label,(subset,rid) in regions.items():
                        incl='✅' if rid in highlight_ids else '❌'
                        st.write(f"**{label}**: {format_set(subset)} {incl}")
            else:
                st.info("Resulting set is empty (∅).")

# --- Power Set Generator Section ---
if show_powerset:
    st.sidebar.subheader("Power Set Settings")
    st.subheader("Power Set Generator")
    ps_input = st.text_input("Enter elements for power set (comma-separated)")
    if ps_input is not None:
        base_set = parse_set(ps_input)
        n = len(base_set)
        if n == 0:
            st.write("Power set of ∅ is: [∅]")
        elif n > 12:
            st.warning(f"Set has {n} elements; power set size is 2^{n}, too large to list.")
            if st.checkbox("Show only power set size", key="ps_count_only"):
                st.write(f"Power set size: {2**n}")
        else:
            sorted_elements = sort_result(base_set)
            power_list = []
            for r in range(n+1):
                for combo in itertools.combinations(sorted_elements, r):
                    power_list.append(set(combo))
            st.write(f"Power set size: {len(power_list)}")
            if st.checkbox("Group subsets by size", key="group_by_size"):
                for size in range(n+1):
                    subsets_size=[s for s in power_list if len(s)==size]
                    formatted=["∅" if not s else "{"+", ".join(str(e) for e in sort_result(s))+"}" for s in subsets_size]
                    st.write(f"**Subsets of size {size} ({len(subsets_size)})**: {formatted}")
            else:
                formatted=["∅" if not s else "{"+", ".join(str(e) for e in sort_result(s))+"}" for s in power_list]
                st.write(formatted)

# --- Set Definition Validator Section ---
if show_validator:
    st.sidebar.subheader("Set Definition Validator")
    user_def = st.text_input("Enter arbitrary set definition or description")
    if user_def:
        if not openai_api_key:
            st.error("Cannot validate via GPT because OPENAI_API_KEY is not configured.")
        else:
            with st.spinner("Interpreting set definition via ChatGPT..."):
                result = interpret_set_definition_via_gpt(user_def)
            note = result.get("note", "")
            if result.get("valid"):
                st.success("Recognized as a valid set definition.")
                st.write("**Type:**", result.get("type"))
                desc = result.get("description")
                if desc:
                    st.write("**Description:**", desc)
                elems = result.get("elements")
                if elems is not None:
                    st.write("**Enumerated Elements:**", format_set(set(elems)))
                    test_val = st.text_input("Test membership: enter a value", key="membership_test")
                    if test_val:
                        tv = parse_element(test_val)
                        if tv in elems:
                            st.success(f"{tv!r} ∈ the set")
                        else:
                            st.info(f"{tv!r} ∉ the set")
                else:
                    st.write("**Elements:** (Infinite or not enumerated)")
                    member_q = st.text_input("Test membership via GPT (enter a value)", key="gpt_member_test")
                    if member_q:
                        if 'Insufficient quota' in note:
                            st.error("Cannot perform membership query: insufficient quota.")
                        else:
                            prompt = (
                                f"Determine if {member_q!r} is an element of the set defined by: {user_def!r}. Respond ONLY with JSON: { 'member': True/False, 'explanation': '...' }."  # noqa: E501
                            )
                            try:
                                resp = openai.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant for set membership queries."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.0,
                                    max_tokens=200,
                                )
                                content = resp.choices[0].message.content.strip()
                                mem_data = json.loads(content)
                                if mem_data.get("member"):
                                    st.success(f"GPT: {member_q!r} ∈ the set. Explanation: {mem_data.get('explanation','')}")
                                else:
                                    st.info(f"GPT: {member_q!r} ∉ the set. Explanation: {mem_data.get('explanation','')}")
                            except Exception as e:
                                st.error(f"Membership query failed: {e}")
            else:
                if 'Insufficient quota' in note:
                    st.error("Not recognized as a valid set: insufficient quota for GPT-based validation. Please check your plan or use local parsing.")
                else:
                    st.error("Not recognized as a valid set: " + note)

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Matplotlib, matplotlib-venn & OpenAI API")
