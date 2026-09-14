"""The evaluation grid (GIST: 4 scopes x 2013-2022) and the prompts of one VLM call.

One call per year: 10 calls per report, each covering the 4 scopes of that year.
Everything here is a constant — the only thing that varies between the 10 calls
is the year, substituted into the two prompt templates below.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Scope = Literal["1", "2mb", "2lb", "3"]

# The gold-standard labels, in the order `collect_` sorts by. The matching
# `YearAnswer` field is the label prefixed with `scope_`, since a field cannot
# start with a digit.
SCOPES: tuple[Scope, ...] = ("1", "2mb", "2lb", "3")

YEARS: tuple[int, ...] = tuple(range(2013, 2023))

# Substituted by `retrieval_query` and `user_prompt`. Not a str.format field on
# purpose: the prompts contain literal JSON braces.
_YEAR = "<YEAR>"


# ---------------------------------------------------------------------------
# Retrieval query
# ---------------------------------------------------------------------------
RETRIEVAL_QUERY = ("Total greenhouse gas emissions in <YEAR>: Scope 1, Scope 2 "
                   "(market-based and location-based) and Scope 3, in tonnes of CO2 equivalent")


def retrieval_query(year: int) -> str:
    return RETRIEVAL_QUERY.replace(_YEAR, str(year))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You extract greenhouse gas (GHG) emission from a corporate sustainability report.
You receive page images of the report. Answer only from what is visible in these images.

Accept a value only if ALL of the following hold:
1. It covers the whole company (group-wide), not a single site, region, or business unit.
2. It is an absolute quantity of CO2 or CO2-equivalent emissions, not an intensity ratio, a percentage change, a reduction, or a target.
3. It is the total for the requested scope, not a subcategory (e.g. not only "business travel" for Scope 3).
4. It is reported for the requested year (a baseline year counts only if that is the year asked for).

Fill all four scope fields. When the images do not show the requested value for a
scope, that scope gets "value": null.
Each field is described in the response schema; follow those descriptions exactly."""

USER_PROMPT = """Report all figures for the year <YEAR>, read from the attached page images:
- scope_1: Scope 1 in <YEAR>
- scope_2mb: Scope 2 market-based in <YEAR>
- scope_2lb: Scope 2 location-based in <YEAR>
- scope_3: Scope 3 in <YEAR>"""


def get_user_prompt(year: int) -> str:
    """Return user prompt str for `year`"""
    return USER_PROMPT.replace(_YEAR, str(year))


# ---------------------------------------------------------------------------
# Answer shape
# ---------------------------------------------------------------------------

class Cell(BaseModel):
    value: Annotated[float | None, Field(description=(
        'The number exactly as printed, without thousands separators and without the '
        'unit. Never estimate, sum, or convert: report 42 for "42 ktCO2eq", not 42000. '
        'null when the requested figure does not appear in the images.'))]

    unit: Annotated[str | None, Field(description=(
        'The unit exactly as printed next to the figure in the image, e.g. "t CO2e", '
        '"ktCO2eq", "million tonnes CO2e". null when value is null, or when no unit '
        'is visible.'))]

    display_type: Annotated[Literal["Table", "Graphic", "Text"] | None, Field(description=(
        'Where the figure is printed: "Table" for a cell of a table, "Graphic" for a '
        'chart, plot or infographic, "Text" for running prose. null when value is null.'))]

    evidence: Annotated[str | None, Field(description=(
        'A short quote, at most 120 characters, of the label and the number as printed, '
        'so the reading can be audited. null when value is null.'))]


class YearAnswer(BaseModel):
    """One VLM call's answer: the four scopes of the year that was asked."""

    scope_1: Annotated[Cell, Field(description=(
        'Scope 1: direct emissions from sources the company owns or controls, for the '
        'year requested in the user message.'))]

    scope_2mb: Annotated[Cell, Field(description=(
        'Scope 2 market-based: indirect emissions from purchased energy, computed with '
        'supplier-specific or contractual emission factors.'))]

    scope_2lb: Annotated[Cell, Field(description=(
        'Scope 2 location-based: indirect emissions from purchased energy, computed with '
        'average grid emission factors. A Scope 2 figure reported without stating its '
        'method belongs here, not in scope_2mb.'))]

    scope_3: Annotated[Cell, Field(description=(
        'Scope 3: the total of all other indirect emissions in the value chain, for the '
        'year requested. A single category, such as business travel or purchased goods, '
        'is not this total.'))]
