from typing import Dict


ALLOWED_SOURCES: Dict[str, Dict[str, str]] = {
    "wikipedia": {
        "domain": "wikipedia.org",
        "desc": "General-purpose encyclopedia: good for definitions, overviews, background context.",
    },
    "github": {
        "domain": "github.com",
        "desc": "Code repositories: good for implementations, libraries, issues, and examples.",
    },
    "reddit": {
        "domain": "reddit.com",
        "desc": "Community discussions: good for practical tips, opinions, debugging threads (verify claims).",
    },
    "arxiv": {
        "domain": "arxiv.org",
        "desc": "Scientific preprints: good for papers, methods, and research context.",
    },
}
