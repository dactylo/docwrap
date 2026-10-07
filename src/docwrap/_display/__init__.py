"""Display formatting -- diff, JSON, summary tables, stats rendering, print orchestration."""

from docwrap._display._diff import iter_diff as iter_diff
from docwrap._display._json import format_json as format_json
from docwrap._display._json import json_error_record as json_error_record
from docwrap._display._print import print_preview as print_preview
from docwrap._display._print import print_summary as print_summary
from docwrap._display._stats import print_stats as print_stats
from docwrap._display._summary import format_summary as format_summary
