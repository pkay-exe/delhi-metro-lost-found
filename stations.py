"""Delhi Metro station network used for the location signal.

The network is a SUBSET of the real system, typed from public knowledge of the line maps, and meant for a
prototype. Check names and order against the official DMRC map before relying on it.

Each station is a node. Two stations on consecutive stops of a line are joined by an edge, and
interchange stations (same name on several lines) join the lines together. "Hops" between two stations
is the number of stops on the shortest route, which is a better location clue than "same city" is.
"""
from collections import deque, defaultdict

# ink = readable text colour on top of the line colour
LINES = {
    "Yellow Line": {"color": "#F5C400", "ink": "#16233B", "segments": [[
        "Samaypur Badli", "Jahangirpuri", "Azadpur", "Model Town", "GTB Nagar", "Vishwavidyalaya",
        "Vidhan Sabha", "Civil Lines", "Kashmere Gate", "Chandni Chowk", "Chawri Bazar", "New Delhi",
        "Rajiv Chowk", "Patel Chowk", "Central Secretariat", "Udyog Bhawan", "Lok Kalyan Marg", "Jor Bagh",
        "Dilli Haat INA", "AIIMS", "Green Park", "Hauz Khas", "Malviya Nagar", "Saket", "Qutab Minar",
        "Chhatarpur", "Sultanpur", "Ghitorni", "Arjan Garh", "Guru Dronacharya", "Sikandarpur", "MG Road",
        "IFFCO Chowk", "HUDA City Centre"]]},
    "Blue Line": {"color": "#1F6FD0", "ink": "#FFFFFF", "segments": [[
        "Dwarka Sector 21", "Dwarka Sector 8", "Dwarka Sector 9", "Dwarka Sector 10", "Dwarka Sector 11",
        "Dwarka Sector 12", "Dwarka Sector 13", "Dwarka Sector 14", "Dwarka", "Dwarka Mor", "Nawada",
        "Uttam Nagar West", "Uttam Nagar East", "Janakpuri West", "Janakpuri East", "Tilak Nagar",
        "Subhash Nagar", "Tagore Garden", "Rajouri Garden", "Ramesh Nagar", "Moti Nagar", "Kirti Nagar",
        "Shadipur", "Patel Nagar", "Rajendra Place", "Karol Bagh", "Jhandewalan", "RK Ashram Marg",
        "Rajiv Chowk", "Barakhamba Road", "Mandi House", "Supreme Court", "Indraprastha", "Yamuna Bank",
        "Akshardham", "Mayur Vihar-I", "Mayur Vihar Extension", "New Ashok Nagar", "Noida Sector 15",
        "Noida Sector 16", "Noida Sector 18", "Botanical Garden", "Golf Course", "Noida City Centre"]]},
    "Red Line": {"color": "#D62839", "ink": "#FFFFFF", "segments": [[
        "Rithala", "Rohini West", "Rohini East", "Pitampura", "Kohat Enclave", "Netaji Subhash Place",
        "Keshav Puram", "Kanhaiya Nagar", "Inderlok", "Shastri Nagar", "Pratap Nagar", "Pul Bangash",
        "Tis Hazari", "Kashmere Gate", "Shastri Park", "Seelampur", "Welcome", "Shahdara", "Mansarovar Park",
        "Jhilmil", "Dilshad Garden", "Shaheed Nagar", "Raj Bagh", "Major Mohit Sharma", "Shyam Park",
        "Mohan Nagar", "Arthala", "Hindon River", "Shaheed Sthal"]]},
    "Green Line": {"color": "#2F9E44", "ink": "#FFFFFF", "segments": [
        ["Kirti Nagar", "Satguru Ram Singh Marg", "Ashok Park Main"],
        ["Inderlok", "Ashok Park Main", "Punjabi Bagh", "Shivaji Park", "Madipur", "Paschim Vihar East",
         "Paschim Vihar West", "Peeragarhi", "Udyog Nagar", "Maharaja Surajmal Stadium", "Nangloi",
         "Nangloi Railway Station", "Rajdhani Park", "Mundka", "Mundka Industrial Area", "Ghevra",
         "Tikri Kalan", "Tikri Border", "Pandit Shree Ram Sharma", "Bahadurgarh City",
         "Brigadier Hoshiyar Singh"]]},
    "Violet Line": {"color": "#7A3FB5", "ink": "#FFFFFF", "segments": [[
        "Kashmere Gate", "Lal Qila", "Jama Masjid", "Delhi Gate", "ITO", "Mandi House", "Janpath",
        "Central Secretariat", "Khan Market", "JLN Stadium", "Jangpura", "Lajpat Nagar", "Moolchand",
        "Kailash Colony", "Nehru Place", "Kalkaji Mandir", "Govind Puri", "Harkesh Nagar Okhla",
        "Jasola Apollo", "Sarita Vihar", "Mohan Estate", "Tughlakabad", "Badarpur Border", "Sarai",
        "NHPC Chowk", "Mewala Maharajpur", "Sector 28 Faridabad", "Badkal Mor", "Old Faridabad",
        "Neelam Chowk Ajronda", "Bata Chowk", "Escorts Mujesar", "Sant Surdas", "Raja Nahar Singh"]]},
    "Pink Line": {"color": "#EE6FA0", "ink": "#16233B", "segments": [[
        "Majlis Park", "Azadpur", "Shalimar Bagh", "Netaji Subhash Place", "Shakurpur", "Punjabi Bagh West",
        "ESI Basaidarapur", "Rajouri Garden", "Mayapuri", "Naraina Vihar", "Delhi Cantt",
        "Durgabai Deshmukh South Campus", "Sir M Vishweshwaraiah Moti Bagh", "Bhikaji Cama Place",
        "Sarojini Nagar", "Dilli Haat INA", "South Extension", "Lajpat Nagar", "Vinobapuri", "Ashram",
        "Sarai Kale Khan Nizamuddin", "Mayur Vihar-I", "Mayur Vihar Pocket 1", "Trilokpuri Sanjay Lake",
        "East Vinod Nagar", "Mandawali", "IP Extension", "Anand Vihar", "Karkarduma", "Karkarduma Court",
        "Krishna Nagar", "East Azad Nagar", "Welcome", "Jaffrabad", "Maujpur Babarpur", "Gokulpuri",
        "Johri Enclave", "Shiv Vihar"]]},
    "Magenta Line": {"color": "#B3236B", "ink": "#FFFFFF", "segments": [[
        "Janakpuri West", "Dabri Mor", "Dashrath Puri", "Palam", "Sadar Bazar Cantonment",
        "Terminal 1 IGI Airport", "Shankar Vihar", "Vasant Vihar", "Munirka", "RK Puram", "IIT Delhi",
        "Hauz Khas", "Panchsheel Park", "Chirag Delhi", "Greater Kailash", "Nehru Enclave", "Kalkaji Mandir",
        "Okhla NSIC", "Sukhdev Vihar", "Jamia Millia Islamia", "Okhla Vihar", "Jasola Vihar Shaheen Bagh",
        "Kalindi Kunj", "Okhla Bird Sanctuary", "Botanical Garden"]]},
    "Airport Express": {"color": "#F28C28", "ink": "#16233B", "segments": [[
        "New Delhi", "Shivaji Stadium", "Dhaula Kuan", "Delhi Aerocity", "IGI Airport",
        "Dwarka Sector 21"]]},
}

MAX_HOPS = 15  # used for unknown or very distant stations

_adj = defaultdict(set)
_edge_line = {}
_lines_of = defaultdict(list)
for _line, _spec in LINES.items():
    for _seg in _spec["segments"]:
        for _s in _seg:
            if _line not in _lines_of[_s]:
                _lines_of[_s].append(_line)
        for _a, _b in zip(_seg, _seg[1:]):
            _adj[_a].add(_b)
            _adj[_b].add(_a)
            _edge_line.setdefault(frozenset((_a, _b)), _line)

STATIONS = sorted(_adj)
_lookup = {" ".join(s.lower().replace("-", " ").split()): s for s in STATIONS}


def normalize_station(text):
    """Map free text to a canonical station name, or None if it is not a known station."""
    key = " ".join(str(text).lower().replace("-", " ").replace(" metro station", "").split())
    return _lookup.get(key)


def lines_of(station):
    return list(_lines_of.get(station, []))


def is_interchange(station):
    return len(_lines_of.get(station, [])) > 1


def _bfs(start):
    dist, parent = {start: 0}, {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in _adj[u]:
            if v not in dist:
                dist[v], parent[v] = dist[u] + 1, u
                q.append(v)
    return dist, parent


_BFS = {s: _bfs(s) for s in STATIONS}
HOPS = {s: _BFS[s][0] for s in STATIONS}


def hops(a, b):
    """Stops on the shortest route between two stations (0 = same station)."""
    if a not in HOPS or b not in HOPS[a]:
        return MAX_HOPS
    return HOPS[a][b]


def path(a, b):
    """Stations on the shortest route from a to b, inclusive."""
    if a not in _BFS or b not in _BFS[a][0]:
        return [a] if a == b else [a, b]
    parent, out, cur = _BFS[a][1], [], b
    while cur is not None:
        out.append(cur)
        cur = parent[cur]
    return out[::-1]


def edge_line(a, b):
    return _edge_line.get(frozenset((a, b)))


if __name__ == "__main__":
    print(len(STATIONS), "stations,", sum(is_interchange(s) for s in STATIONS), "interchanges")
    print("Hauz Khas -> Green Park:", hops("Hauz Khas", "Green Park"))
    print("Hauz Khas -> Kashmere Gate:", hops("Hauz Khas", "Kashmere Gate"), path("Hauz Khas", "Kashmere Gate"))
    print("max hops from Rithala:", max(HOPS["Rithala"].values()), "| connected:", len(HOPS["Rithala"]) == len(STATIONS))
