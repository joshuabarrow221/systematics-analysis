"""Print effective physical bin edges and profile centers from the YAML file."""
import argparse
import json
from pathlib import Path

import numpy as np

from .v2_config import load_config, resolve_edges, OBSERVABLE_IDS, SELECTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/analysis_v2.yaml")
    parser.add_argument("--profile", help="Default: every configured binning profile")
    args = parser.parse_args()
    config = load_config(args.config)
    profiles = [args.profile] if args.profile else config["binning_profiles"]
    # Print each distinct edge array once; name every selection that uses it.
    distinct = {}
    unavailable = []
    for profile in profiles:
        for containment in ("all", "fc", "pc"):
            for selection in SELECTIONS:
                for observable in OBSERVABLE_IDS:
                    edges, note = resolve_edges(config, profile, containment, selection, observable)
                    name = f"{profile}/{containment}/{selection['id']}/{observable}"
                    if edges is None:
                        unavailable.append({"selection": name, "reason": note})
                        continue
                    key = (tuple(edges), observable == "angular_resolution")
                    if key not in distinct:
                        distinct[key] = {"used_by": [], "edges": edges.tolist()}
                        if key[1]:
                            centers = (np.sqrt(edges[:-1] * edges[1:]) if np.all(edges > 0)
                                       else (edges[:-1] + edges[1:]) / 2)
                            distinct[key]["profile_centers"] = centers.tolist()
                    distinct[key]["used_by"].append(name)
    print(json.dumps({"config_sha256": config["_sha256"],
                      "binning": list(distinct.values()), "unavailable": unavailable}, indent=2))


if __name__ == "__main__":
    main()
