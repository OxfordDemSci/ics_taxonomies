import os
import re
import csv
import ast
import sys
import requests
import spacy.util
import subprocess
import unicodedata
import numpy as np
import pandas as pd
from textblob import TextBlob
from textstat import textstat
from markdown import markdown
from bs4 import BeautifulSoup
from get_openalex_data import get_openalex_data
from get_dimensions_data import get_dimensions_data, make_paper_level

from pathlib import Path

# ===============================
# Utilities and pipeline helpers
# ===============================

def log_row_count(func):
    """Decorator to log the number of rows in a DataFrame after applying a function."""
    def wrapper(df, *args, **kwargs):
        result = func(df, *args, **kwargs)
        print(f"Number of rows after {func.__name__}: {len(result)}")
        return result
    return wrapper


def get_impact_data(raw_path):
    """Download ICS data + tags to raw_path (unrelated to results sheet)."""
    print('Getting ICS Data!')
    url = "https://results2021.ref.ac.uk/impact/export-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / 'raw_ref_ics_data.xlsx', 'wb').write(r.content)

    url = "https://results2021.ref.ac.uk/impact/export-tags-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / 'raw_ref_ics_tags_data.xlsx', 'wb').write(r.content)


def get_environmental_data(raw_path):
    """Download Environmental data to raw_path."""
    print('Getting Environmental Data!')
    url = "https://results2021.ref.ac.uk/environment/export-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / 'raw_ref_environment_data.xlsx', 'wb').write(r.content)


def get_all_results(raw_path):
    """
    Ensure the REF results workbook exists (XLSX). If missing, download it.
    """
    xlsx_path = raw_path / 'raw_ref_results_data.xlsx'
    if xlsx_path.exists():
        print("Results XLSX present locally; not downloading.")
        return
    print('Getting Results Data (XLSX)…')
    url = "https://results2021.ref.ac.uk/profiles/export-all"
    r = requests.get(url, allow_redirects=True)
    open(xlsx_path, 'wb').write(r.content)


def get_output_data(raw_path):
    """Download Outputs data to raw_path."""
    print('Getting Outputs Data!')
    url = "https://results2021.ref.ac.uk/outputs/export-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / 'raw_ref_outputs_data.xlsx', 'wb').write(r.content)


def check_id_overlap(a, b):
    """Quick overlap diagnostics between two ID lists."""
    print('Checking ID Overlap!')
    print("{} of elements in B present in A".format(np.mean([i in a for i in b])))
    print("{} of elements in A present in B".format(np.mean([i in b for i in a])))
    print("{} missing in A but present in B".format([i for i in a if i not in b]))
    print("{} missing in B but present in A".format([i for i in b if i not in a]))


def format_ids(df):
    """
    Standardise institution + UoA identifiers and build 'uoa_id'.
    Expects either 'Institution UKPRN code' or 'Institution code (UKPRN)'.
    """
    if 'Institution UKPRN code' in df.columns:
        df = df.rename(columns={'Institution UKPRN code': 'inst_id'})
    if 'Institution code (UKPRN)' in df.columns:
        df = df.rename(columns={'Institution code (UKPRN)': 'inst_id'})
    df = df[df['inst_id'] != ' ']
    df = df.astype({'inst_id': 'int'})
    df['uoa_id'] = df['Unit of assessment number'].astype(int).astype(str) + \
                   df['Multiple submission letter'].fillna('').astype(str)
    return df


def merge_ins_uoa(df1, df2, id1='inst_id', id2='uoa_id'):
    """Left-merge df2 into df1 on inst_id and uoa_id with key assertions."""
    assert all(df1[id1].isin(df2[id1]))
    assert all(df1[id2].isin(df2[id2]))
    return df1.merge(df2, how='left', on=[id1, id2])


@log_row_count
def clean_ics_level(raw_path, edit_path):
    """Clean ICS-level data and persist a cleaned Excel."""
    print('Cleaning ICS Level Data!')
    raw_ics = pd.read_excel(raw_path / 'raw_ref_ics_data.xlsx', engine="openpyxl")
    raw_ics['Title'] = raw_ics['Title'].apply(
        lambda val: unicodedata.normalize('NFKD', str(val)).encode('ascii', 'ignore').decode()
    )
    raw_ics = format_ids(raw_ics)
    raw_ics.to_excel(edit_path / 'clean_ref_ics_data.xlsx', index=False)
    return raw_ics


# ---------- Results reader (Excel) ----------

def read_results_table(raw_path: Path, sheet=0) -> pd.DataFrame:
    """
    Load the REF results workbook by sheet using pandas.read_excel.
    Assumptions:
      - The header row is at Excel row 7 (0-index header=6).
      - We keep everything as object dtype to avoid coercion.
    """
    xlsx_path = raw_path / "raw_ref_results_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Missing {xlsx_path}. Download step should have created it."
        )
    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet,
        header=6,          # header row starts at line 7 in the workbook
        engine="openpyxl",
        dtype=object
    )
    # normalise whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_dep_level(raw_path, edit_path):
    """
    Build a wide department-level scorecard and merge with environmental metrics.
    Results table is loaded from the original Excel workbook.
    """
    print('Cleaning Department Level Data!')

    # Read results (Excel)
    raw_results = read_results_table(raw_path, sheet=0)

    # Normalise institution code variants
    raw_results = raw_results.rename(columns={
        'Institution UKPRN code': 'inst_id',
        'Institution code (UKPRN)': 'inst_id'
    })

    raw_results = format_ids(raw_results)
    raw_results = raw_results.rename(
        columns={'FTE of submitted staff': 'fte',
                 '% of eligible staff submitted': 'fte_pc'}
    )

    # Build wide score card by institution and uoa_id
    score_types = ['4*', '3*', '2*', '1*', 'Unclassified']
    wide_score_card = pd.pivot_table(
        raw_results[['inst_id', 'uoa_id', 'Profile'] + score_types],
        index=['inst_id', 'uoa_id'],
        columns='Profile',
        values=score_types,
        aggfunc='first'
    )
    # Flatten MultiIndex columns like ('4*','Impact') -> '4*_Impact'
    wide_score_card.columns = wide_score_card.columns.map('_'.join)
    wide_score_card = wide_score_card.reset_index()

    # Environmental data
    raw_env_path = raw_path / 'raw_ref_environment_data.xlsx'

    # Doctoral degrees
    raw_env_doctoral = pd.read_excel(raw_env_path, sheet_name="ResearchDoctoralDegreesAwarded",
                                     skiprows=4, engine="openpyxl")
    raw_env_doctoral = format_ids(raw_env_doctoral)
    number_cols = [c for c in raw_env_doctoral.columns if 'Number of doctoral' in c]
    raw_env_doctoral['num_doc_degrees_total'] = raw_env_doctoral[number_cols].sum(axis=1)

    # Research income
    raw_env_income = pd.read_excel(raw_env_path, sheet_name="ResearchIncome",
                                   skiprows=4, engine="openpyxl")
    raw_env_income = format_ids(raw_env_income)
    raw_env_income = raw_env_income.rename(columns={
        'Average income for academic years 2013-14 to 2019-20': 'av_income',
        'Total income for academic years 2013-14 to 2019-20': 'tot_income'
    })
    tot_inc = raw_env_income[raw_env_income['Income source'] == 'Total income']

    # Research income in-kind
    raw_env_income_inkind = pd.read_excel(raw_env_path, sheet_name="ResearchIncomeInKind",
                                          skiprows=4, engine="openpyxl")
    raw_env_income_inkind = format_ids(raw_env_income_inkind)
    raw_env_income_inkind = raw_env_income_inkind.rename(
        columns={'Total income for academic years 2013-14 to 2019-20': 'tot_inc_kind'}
    )
    tot_inc_kind = raw_env_income_inkind.loc[
        raw_env_income_inkind['Income source'] == 'Total income-in-kind'
    ]

    # Merge all dept-level data
    raw_dep = merge_ins_uoa(
        raw_results[['inst_id', 'uoa_id', 'fte', 'fte_pc']].drop_duplicates(),
        wide_score_card
    )
    raw_dep = merge_ins_uoa(
        raw_dep, raw_env_doctoral[['inst_id', 'uoa_id', 'num_doc_degrees_total']]
    )
    raw_dep = merge_ins_uoa(
        raw_dep, tot_inc[['inst_id', 'uoa_id', 'av_income', 'tot_income']]
    )
    raw_dep = merge_ins_uoa(
        raw_dep, tot_inc_kind[['inst_id', 'uoa_id', 'tot_inc_kind']]
    )
    raw_dep.to_excel(edit_path / 'clean_ref_dep_data.xlsx', index=False)


def markdown_to_text(text):
    """Convert Markdown-formatted text to plain text."""
    html = markdown(text)
    bs = BeautifulSoup(html, features="html.parser")
    return bs.get_text()


def get_urls(text):
    """Extract URLs and DOIs from Markdown-formatted text."""
    urls = re.findall(r"\[.*\]\((.*)\)", markdown_to_text(text))
    doi_urls = [u for u in urls if "doi.org" in u]
    return len(urls), urls, len(doi_urls), doi_urls


def prepare_spacy():
    """Load spaCy language model; download if needed."""
    model_name = "en_core_web_lg"
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    return spacy.load(model_name)


def count_noun_verb_phrases(text):
    """Count noun phrases and verb tokens in text."""
    nlp = prepare_spacy()
    doc = nlp(markdown_to_text(text))
    noun_phrase_count = sum(1 for _ in doc.noun_chunks)
    verb_phrase_count = sum(1 for t in doc if t.pos_ == "VERB")
    return noun_phrase_count, verb_phrase_count


def gen_readability_scores(df, edit_path, section_columns):
    """Compute Flesch scores by section and overall; write CSV."""
    output_path = edit_path / 'ics_readability.csv'
    print('Calculating Readability Scores!')
    for i, s in zip(range(1, 6), section_columns):
        df[f"s{i}_flesch_score"] = df[s].apply(lambda x: textstat.flesch_reading_ease(markdown_to_text(x)))
    df["flesch_score"] = df.apply(
        lambda x: textstat.flesch_reading_ease("\n".join([x[s] for s in section_columns])),
        axis=1,
    )
    cols = [f"s{i}_flesch_score" for i in range(1, 6)] + ['flesch_score', 'REF impact case study identifier']
    print(f"Writing readability scores to {output_path}")
    df[cols].to_csv(output_path, index=False)


@log_row_count
def get_readability_scores(df, edit_path):
    """Load readability scores and merge."""
    file_path = edit_path / 'ics_readability.csv'
    print('Loading Readability Scores!')
    readability = pd.read_csv(file_path, index_col=None)
    assert len(df) == len(readability)
    key = 'REF impact case study identifier'
    assert set(df[key]) == set(readability[key])
    return pd.merge(df, readability, on=[key], how='left')


def gen_pos_features(df, edit_path, section_columns):
    """Compute POS features by section; write CSV."""
    output_path = edit_path / "ics_pos_features.csv"
    print('Calculating POS Features')
    for i, s in zip(range(1, 6), section_columns):
        df[[f"s{i}_np_count", f"s{i}_vp_count"]] = df[s].apply(lambda x: pd.Series(count_noun_verb_phrases(x)))
    cols = [f"s{i}_np_count" for i in range(1, 6)] + [f"s{i}_vp_count" for i in range(1, 6)] + \
           ['REF impact case study identifier']
    print(f"Writing pos features to {output_path}")
    df[cols].to_csv(output_path, index=False)


@log_row_count
def get_pos_features(df, edit_path):
    """Load POS features and merge."""
    file_path = edit_path / 'ics_pos_features.csv'
    print('Loading pos features!')
    pos_features = pd.read_csv(file_path, index_col=None)
    assert len(df) == len(pos_features)
    key = 'REF impact case study identifier'
    assert set(df[key]) == set(pos_features[key])
    return pd.merge(df, pos_features, on=[key], how='left')


def get_sentiment_score(text):
    """TextBlob polarity in [-1, 1]."""
    plain_text = markdown_to_text(text)
    blob = TextBlob(plain_text)
    return blob.sentiment_assessments.polarity


def gen_sentiment_scores(df, edit_path, section_columns):
    """Compute sentiment per section and overall; write CSV."""
    output_path = edit_path / "ics_sentiment_scores.csv"
    print('Calculating Sentiment Scores!')
    for i, s in zip(range(1, 6), section_columns):
        df[f"s{i}_sentiment_score"] = df[s].apply(get_sentiment_score)
    df["sentiment_score"] = df.apply(
        lambda row: get_sentiment_score("\n".join([row[s] for s in section_columns])), axis=1
    )
    cols = [f"s{i}_sentiment_score" for i in range(1, 6)] + ["sentiment_score", "REF impact case study identifier"]
    print(f"Writing sentiment scores to {output_path}")
    df[cols].to_csv(output_path, index=False)


@log_row_count
def get_sentiment_scores(df, edit_path):
    """Load sentiment scores and merge."""
    file_path = edit_path / 'ics_sentiment_scores.csv'
    print('Loading sentiment scores!')
    sentiment_scores = pd.read_csv(file_path, index_col=None)
    assert len(df) == len(sentiment_scores)
    key = 'REF impact case study identifier'
    assert set(df[key]) == set(sentiment_scores[key])
    return pd.merge(df, sentiment_scores, on=[key], how='left')


@log_row_count
def add_postcodes(df, sup_path):
    """Add postcode info and derive district/area."""
    print('Add some postcodes!')
    postcodes = pd.read_csv(sup_path / 'ukprn_postcode.csv')
    df = pd.merge(df, postcodes, how='left', on='inst_id')
    dist = 'inst_postcode_district'
    area = 'inst_postcode_area'
    df[dist] = df['Post Code'].apply(lambda x: x.split(' ')[0])
    df[area] = df[dist].apply(lambda x: x[0:re.search(r"\d", x).start()])
    return df


@log_row_count
def add_url(df):
    """Add per-ICS URL."""
    field = 'REF impact case study identifier'
    base = 'https://results2021.ref.ac.uk/impact/'
    df['ics_url'] = df[field].apply(lambda x: base + x + '?page=1')
    return df


def get_paths(data_path):
    """Build standard folder paths."""
    data_path = Path(data_path)
    raw_path = data_path / 'raw'
    edit_path = data_path / 'edit'
    sup_path = data_path / 'supplementary'
    manual_path = data_path / 'manual'
    final_path = data_path / 'final'
    topic_path = data_path / 'reassignments'
    dim_path = data_path / 'dimensions_returns'
    openalex_path = data_path / 'openalex_returns'
    ics_staff_rows_path = data_path / 'ics_staff_rows'
    ics_grants_path = data_path / 'ics_grants'
    return (raw_path, edit_path, sup_path, manual_path,
            final_path, topic_path, dim_path, openalex_path,
            ics_staff_rows_path,
            ics_grants_path)


def make_region_country_list(regions, country_groups):
    """Return countries from region(s) provided."""
    result = float('NaN')
    if isinstance(regions, str):
        region_list = regions.split('; ')
        i_country_groups = country_groups['Union/region'].isin(region_list)
        country_list = list(set(country_groups[i_country_groups]['ISO3 codes']))
        result = '; '.join(sorted(country_list))
    return result


def clean_country_strings(countries):
    """Sort/remove duplicates/clean a semicolon-separated ISO3 string."""
    result = float('NaN')
    if isinstance(countries, str):
        countries_list = list(sorted(set(countries.split('; '))))
        countries_list = [x for x in countries_list if 'nan' not in x]
        result = '; '.join(countries_list)
    return result


def make_countries_file(manual_path):
    """Construct countries DataFrame from manual workbook."""
    df = pd.read_excel(os.path.join(manual_path, 'funder_countries', 'funders_countries_lookup.xlsx'),
                       sheet_name='funders_countries_lookup', engine='openpyxl')
    country_groups = pd.read_excel(os.path.join(manual_path, 'funder_countries', 'funders_countries_lookup.xlsx'),
                                   sheet_name='country_groups', engine='openpyxl')
    for var in ['suggested_Countries[alpha-3]_change',
                'suggested_union_change',
                'suggested_Countries[union]_change',
                'suggested_region_change',
                'suggested_Countries[region]_change']:
        df[var] = df[var].str.strip()

    df['region_extracted'] = np.where(df['suggested_region_change'].isnull(),
                                      df['region'], df['suggested_region_change'])
    df['countries_region_extracted'] = [
        make_region_country_list(regions=x, country_groups=country_groups) for x in list(df['region_extracted'])
    ]

    df['union_extracted'] = np.where(df['suggested_union_change'].isnull(),
                                     df['union'], df['suggested_union_change'])
    df['countries_union_extracted'] = [
        make_region_country_list(regions=x, country_groups=country_groups) for x in list(df['union_extracted'])
    ]

    df['countries_specific_extracted'] = np.where(
        df['suggested_Countries[alpha-3]_change'].isnull(),
        df['Countries[alpha-3]'],
        df['suggested_Countries[alpha-3]_change']
    )

    cols_to_combine = ['countries_specific_extracted', 'countries_union_extracted', 'countries_region_extracted']
    df['countries_extracted'] = df[cols_to_combine].apply(lambda row: '; '.join(row.values.astype('str')), axis=1)

    df['countries_extracted'] = df['countries_extracted'].apply(clean_country_strings)
    df['countries_specific_extracted'] = df['countries_specific_extracted'].apply(clean_country_strings)
    df['countries_union_extracted'] = df['countries_union_extracted'].apply(clean_country_strings)
    df['countries_region_extracted'] = df['countries_region_extracted'].apply(clean_country_strings)

    df = df.dropna(subset=['countries_extracted', 'union_extracted', 'region_extracted'], how='all')
    return df[['REF impact case study identifier',
               'countries_specific_extracted',
               'region_extracted',
               'countries_region_extracted',
               'union_extracted',
               'countries_union_extracted',
               'countries_extracted']]


@log_row_count
def load_country(df, manual_path):
    """Merge country data."""
    print('Loading clean country data')
    country = make_countries_file(manual_path)
    return pd.merge(df, country, how='left', on='REF impact case study identifier')


def make_funder_file(manual_path):
    """Construct funder DataFrame and normalise abbreviations."""
    df = pd.read_excel(os.path.join(manual_path, 'funder_countries', 'funders_countries_lookup.xlsx'),
                       sheet_name='funders_countries_lookup', engine='openpyxl')
    var = 'suggested_CountriesFunders[full name]_change'
    df = df[['REF impact case study identifier', var]]
    df[var] = df[var].str.strip()
    df[var] = np.where(((df[var].str.len() < 2) | df[var].isnull()), np.nan, df[var])
    df = df[df[var].notnull()]
    df = df.rename({var: 'funders_extracted'}, axis=1)
    df['funders_extracted'] = df['funders_extracted'].str.replace('EURC', 'EC')
    df['funders_extracted'] = df['funders_extracted'].str.replace('NIHCR', 'NIHR')
    df['funders_extracted'] = df['funders_extracted'].str.replace('LHT', 'LT')
    df['funders_extracted'] = df['funders_extracted'].str.replace('WCT', 'WT')
    return df[['REF impact case study identifier', 'funders_extracted']]


@log_row_count
def load_funder(df, manual_path):
    """Merge funder data."""
    print('Loading clean funder data')
    funder = make_funder_file(manual_path)
    return pd.merge(df, funder, how='left', on='REF impact case study identifier')


@log_row_count
def load_dept_vars(df, edit_path):
    """Load department vars, compute GPAs, and merge."""
    print('Loading department variables')
    dept_vars = pd.read_excel(edit_path / 'clean_ref_dep_data.xlsx', engine='openpyxl')

    dept_vars['ICS_GPA'] = (pd.to_numeric(dept_vars['4*_Impact'], errors='coerce') * 4 +
                            pd.to_numeric(dept_vars['3*_Impact'], errors='coerce') * 3 +
                            pd.to_numeric(dept_vars['2*_Impact'], errors='coerce') * 2 +
                            pd.to_numeric(dept_vars['1*_Impact'], errors='coerce')) / 100
    dept_vars['Environment_GPA'] = (pd.to_numeric(dept_vars['4*_Environment'], errors='coerce') * 4 +
                                    pd.to_numeric(dept_vars['3*_Environment'], errors='coerce') * 3 +
                                    pd.to_numeric(dept_vars['2*_Environment'], errors='coerce') * 2 +
                                    pd.to_numeric(dept_vars['1*_Environment'], errors='coerce')) / 100
    dept_vars['Output_GPA'] = (pd.to_numeric(dept_vars['4*_Outputs'], errors='coerce') * 4 +
                               pd.to_numeric(dept_vars['3*_Outputs'], errors='coerce') * 3 +
                               pd.to_numeric(dept_vars['2*_Outputs'], errors='coerce') * 2 +
                               pd.to_numeric(dept_vars['1*_Outputs'], errors='coerce')) / 100
    dept_vars['Overall_GPA'] = (pd.to_numeric(dept_vars['4*_Overall'], errors='coerce') * 4 +
                                pd.to_numeric(dept_vars['3*_Overall'], errors='coerce') * 3 +
                                pd.to_numeric(dept_vars['2*_Overall'], errors='coerce') * 2 +
                                pd.to_numeric(dept_vars['1*_Overall'], errors='coerce')) / 100

    cols = ['inst_id', 'uoa_id', 'fte', 'num_doc_degrees_total', 'av_income', 'tot_income',
            'tot_inc_kind', 'ICS_GPA', 'Environment_GPA', 'Output_GPA', 'Overall_GPA']

    return pd.merge(df, dept_vars[cols], how='left',
                    left_on=['inst_id', 'uoa_id'], right_on=['inst_id', 'uoa_id']).drop('uoa_id', axis=1)


@log_row_count
def load_topic_data(df, manual_path, topic_path):
    """Merge topic modelling data if present."""
    print('Loading topic data')
    topic_model_path = topic_path / 'reassignments_final.csv'
    if not topic_model_path.exists():
        print(f"File not found: {topic_model_path}")
        print("WARNING: Not including topic model data columns.")
        print("Returning original file")
        return df

    vars = [
        'REF impact case study identifier',
        'topic1', 'topic1_fit', 'topic1_reassign', 'topic1_theme', 'topic1_theme_long',
        'topic2', 'topic2_fit', 'topic2_reassign', 'topic2_theme', 'topic2_theme_long',
        'topic3', 'topic3_fit', 'topic3_reassign', 'topic3_theme', 'topic3_theme_long',
        'topic_llm', 'bert_prob',
        'topic1_name', 'topic1_name_long',
        'topic2_name', 'topic2_name_long',
        'topic3_name', 'topic3_name_long'
    ]

    topic_model = pd.read_csv(topic_model_path, usecols=vars, index_col=None)
    merged = pd.merge(df, topic_model, how='left', on='REF impact case study identifier')
    return merged


@log_row_count
def make_and_load_tags(df, raw_path, edit_path):
    """Process tag data and merge."""
    print('Loading and merging tags!')
    tags = pd.read_excel(raw_path / 'raw_ref_ics_tags_data.xlsx',
                         sheet_name='Sheet1',
                         skiprows=4,
                         usecols=['REF case study identifier', 'Tag type', 'Tag identifier', 'Tag value', 'Tag group'],
                         engine='openpyxl')
    tags = tags[tags['REF case study identifier'].notnull()]

    with open(edit_path / 'clean_ref_ics_tags_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['REF case study identifier',
                         'Underpinning research subject tag values',
                         'Underpinning research subject tag group',
                         'UK Region tag values',
                         'UK Region tag group'])
        for ics in tags['REF case study identifier'].unique():
            temp = tags[tags['REF case study identifier'] == ics]
            research = temp[temp['Tag type'] == 'Underpinning research subject']
            region = temp[temp['Tag type'] == 'UK Region']
            writer.writerow([
                ics,
                list(research['Tag value']), list(research['Tag group']),
                list(region['Tag value']), list(region['Tag group'])
            ])

    tags = pd.read_csv(edit_path / 'clean_ref_ics_tags_data.csv')
    return pd.merge(df, tags, how='left',
                    left_on='REF impact case study identifier',
                    right_on='REF case study identifier').drop('REF case study identifier', axis=1)


@log_row_count
def load_scientometric_data(df, dim_path, openalex_path):
    """Merge Dimensions/OpenAlex scientometric data if present."""
    print("Loading scientometric data")
    fp_dim = os.path.join(dim_path, 'merged_dimensions.xlsx')
    if not os.path.exists(fp_dim):
        print(f"File not found: {fp_dim}")
        print("WARNING: Not including Dimensions scientometric data columns.")
        print("Returning original file")
        return df
    dim_df = pd.read_excel(fp_dim, engine="openpyxl")

    fp_oa = os.path.join(openalex_path, 'merged_openalex.csv')
    if not os.path.exists(fp_oa):
        print(f"File not found: {fp_oa}")
        print("WARNING: Not including OpenAlex scientometric data columns.")
        print("Returning original file")
        return df
    openalex_df = pd.read_csv(fp_oa)

    merger = pd.DataFrame(columns=['REF impact case study identifier', 'scientometric_data'])
    idx = 0
    outer_dict = {'Dimensions': {}, 'OpenAlex': {}}

    for ics in dim_df['Key'].unique():
        counter = 0
        mydict = {}
        ics_df = dim_df[dim_df['Key'] == ics]
        for index in ics_df.index:
            paper = f"Research_{counter}"
            mydict[paper] = {}
            for key, value in {'dimensions_id': 'id',
                               'title_preferred': 'preferred',
                               'field_of_research': 'category_for',
                               'unit of assessment': 'category_uoa',
                               'type': 'type',
                               'concepts': 'concepts',
                               'date': 'date_normal',
                               'times_cited': 'Times Cited',
                               'recent_citations': 'Recent Citations',
                               'field_citation_ratio': 'Field Citation Ratio',
                               'relative_citation_ratio': 'Relative Citation Ratio',
                               'altmetric': 'Altmetric',
                               'open_access': 'open_access_categories_v2',
                               'researcher_cities': 'research_org_cities',
                               'researcher_countries': 'research_org_countries'}.items():
                try:
                    mydict[paper][key] = ics_df.at[index, value]
                except ValueError:
                    pass
            try:
                mydict[paper]['journal'] = ast.literal_eval(ics_df.at[index, 'journal'])['title']
            except Exception:
                pass
            try:
                paper_fors = ics_df.at[index, 'category_for']
                if paper_fors is not np.nan:
                    first_level = paper_fors.split('second_level')[0]
                    l1 = re.findall(r"'name': '(.*?)'", first_level)
                    second_level = paper_fors.split('second_level')[1]
                    l2 = re.findall(r"'name': '(.*?)'", second_level)
                    mydict[paper]['L1_ANZSRC_FoR'] = l1
                    mydict[paper]['L2_ANZSRC_FoR'] = l2
            except Exception:
                pass
            try:
                paper_uoas = ics_df.at[index, 'category_uoa']
                if paper_uoas is not np.nan:
                    uoas = paper_uoas.split('second_level')[0]
                    uoas = re.findall(r"'name': '(.*?)'", uoas)
                    mydict[paper]['Category_UoA'] = uoas
            except Exception:
                pass
            counter += 1
        outer_dict['Dimensions'] = mydict

        counter = 0
        mydict = {}
        ics_df = openalex_df[openalex_df['Key'] == ics]
        for index in ics_df.index:
            paper = f"Research_{counter}"
            try:
                mydict[paper] = ast.literal_eval(openalex_df.at[index, 'OpenAlex_Response'])
            except Exception:
                mydict[paper] = openalex_df.at[index, 'OpenAlex_Response']
            counter += 1
        outer_dict['OpenAlex'] = mydict

        merger.at[idx, 'REF impact case study identifier'] = ics
        merger.at[idx, 'scientometric_data'] = outer_dict
        idx += 1

    return pd.merge(df, merger, how='left', on='REF impact case study identifier')


def return_dim_id(path, filename):
    """Load and return the project ID from a file."""
    with open(path / filename, "r") as file:
        return file.readline().strip()


def get_ics_staff_rows(df, ics_staff_rows_path):
    if os.path.join(ics_staff_rows_path, 'ICS_aggregated_staff_rows.csv') is False:
        print('No staff rows data! Run ref_staff.py')
        return df
    else:
        staff_rows = pd.read_csv(os.path.join(ics_staff_rows_path, 'ICS_aggregated_staff_rows.csv'))
        print(f"Were missing {len(df)-len(staff_rows)} staff rows! investigate!")
        return pd.merge(df, staff_rows, how='left', on='REF impact case study identifier')


def get_ics_grants(df, ics_grants_path):
    if os.path.join(ics_grants_path, 'ICS_grants_aggregated.csv') is False:
        print('No ICS grants data! Run get_ics_grants.py')
        return df
    else:
        grants_rows = pd.read_csv(os.path.join(ics_grants_path, 'ICS_grants_aggregated.csv'))
        print(f"Were missing {len(df)-len(grants_rows)} grants rows! investigate!")
        return pd.merge(df, grants_rows, how='left', on='REF impact case study identifier')


def get_university_class(df, manual_path):
    if os.path.join(manual_path,
        'university_category',
        'ref_unique_institutions.csv') is False:
        print('No university classifications file! Go make one?!')
        return df
    else:
        university_class = pd.read_csv(os.path.join(manual_path,
            'university_category',
            'ref_unique_institutions.csv'))
        print("Merged in university classifications data.")
        return pd.merge(df, university_class, how='left', on='Institution name')


def get_panel_and_UoA_names(df):
    import pandas as pd
    mapping = [
        {'Unit of assessment number':  1, 'Unit of assessment': 'Clinical Medicine',
         'Main Panel': 'A'},
        {'Unit of assessment number':  2, 'Unit of assessment': 'Public Health, Health Services and Primary Care',
         'Main Panel': 'A'},
        {'Unit of assessment number':  3, 'Unit of assessment': 'Allied Health Professions, Dentistry, Nursing and Pharmacy',
         'Main Panel': 'A'},
        {'Unit of assessment number':  4, 'Unit of assessment': 'Psychology, Psychiatry and Neuroscience',
         'Main Panel': 'A'},
        {'Unit of assessment number':  5, 'Unit of assessment': 'Biological Sciences',
         'Main Panel': 'A'},
        {'Unit of assessment number':  6, 'Unit of assessment': 'Agriculture, Food and Veterinary Sciences',
         'Main Panel': 'A'},
        {'Unit of assessment number':  7, 'Unit of assessment': 'Earth Systems and Environmental Sciences',
         'Main Panel': 'B'},
        {'Unit of assessment number':  8, 'Unit of assessment': 'Chemistry',
         'Main Panel': 'B'},
        {'Unit of assessment number':  9, 'Unit of assessment': 'Physics',
         'Main Panel': 'B'},
        {'Unit of assessment number': 10, 'Unit of assessment': 'Mathematical Sciences',
         'Main Panel': 'B'},
        {'Unit of assessment number': 11, 'Unit of assessment': 'Computer Science and Informatics',
         'Main Panel': 'B'},
        {'Unit of assessment number': 12, 'Unit of assessment': 'Engineering',
         'Main Panel': 'B'},
        {'Unit of assessment number': 13, 'Unit of assessment': 'Architecture, Built Environment and Planning',
         'Main Panel': 'C'},
        {'Unit of assessment number': 14, 'Unit of assessment': 'Geography and Environmental Studies',
         'Main Panel': 'C'},
        {'Unit of assessment number': 15, 'Unit of assessment': 'Archaeology',
         'Main Panel': 'C'},
        {'Unit of assessment number': 16, 'Unit of assessment': 'Economics and Econometrics',
         'Main Panel': 'C'},
        {'Unit of assessment number': 17, 'Unit of assessment': 'Business and Management Studies',
         'Main Panel': 'C'},
        {'Unit of assessment number': 18, 'Unit of assessment': 'Law',
         'Main Panel': 'C'},
        {'Unit of assessment number': 19, 'Unit of assessment': 'Politics and International Studies',
         'Main Panel': 'C'},
        {'Unit of assessment number': 20, 'Unit of assessment': 'Social Work and Social Policy',
         'Main Panel': 'C'},
        {'Unit of assessment number': 21, 'Unit of assessment': 'Sociology',
         'Main Panel': 'C'},
        {'Unit of assessment number': 22, 'Unit of assessment': 'Anthropology and Development Studies',
         'Main Panel': 'C'},
        {'Unit of assessment number': 23, 'Unit of assessment': 'Education',
         'Main Panel': 'C'},
        {'Unit of assessment number': 24, 'Unit of assessment': 'Sport and Exercise Sciences, Leisure and Tourism',
         'Main Panel': 'C'},
        {'Unit of assessment number': 25, 'Unit of assessment': 'Area Studies',
         'Main Panel': 'D'},
        {'Unit of assessment number': 26, 'Unit of assessment': 'Modern Languages and Linguistics',
         'Main Panel': 'D'},
        {'Unit of assessment number': 27, 'Unit of assessment': 'English Language and Literature',
         'Main Panel': 'D'},
        {'Unit of assessment number': 28, 'Unit of assessment': 'History',
         'Main Panel': 'D'},
        {'Unit of assessment number': 29, 'Unit of assessment': 'Classics',
         'Main Panel': 'D'},
        {'Unit of assessment number': 30, 'Unit of assessment': 'Philosophy',
         'Main Panel': 'D'},
        {'Unit of assessment number': 31, 'Unit of assessment': 'Theology and Religious Studies',
         'Main Panel': 'D'},
        {'Unit of assessment number': 32, 'Unit of assessment': 'Art and Design: History, Practice and Theory',
         'Main Panel': 'D'},
        {'Unit of assessment number': 33, 'Unit of assessment': 'Music, Drama, Dance, Performing Arts, Film and Screen Studies',
         'Main Panel': 'D'},
        {'Unit of assessment number': 34, 'Unit of assessment': 'Communication, Cultural and Media Studies, Library and Information Management',
         'Main Panel': 'D'}
    ]
    M = pd.DataFrame(mapping)
    print('Merged on UoA and Panel Names')
    return df.merge(
        M,
        on='Unit of assessment number',
        how='left',
        validate='many_to_one'  # ensures each key in Panel matches at most one in M
    )

def get_thematic_indicators(df):
    # --- 1) Choose the two source columns ---
    COLS = ['1. Summary of the impact', '4. Details of the impact']

    # --- 2) Normalise and concatenate text safely (handles NaNs, dashes, spacing) ---
    def _normalise_text(s: pd.Series) -> pd.Series:
        # Ensure string, normalise Unicode dashes to '-', collapse whitespace
        s = s.fillna('').astype(str)
        s = s.str.replace('[\u2012\u2013\u2014\u2015]', '-', regex=True)  # en/em dashes → '-'
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
        return s

    norm = _normalise_text(df[COLS[0]]) + ' ' + _normalise_text(df[COLS[1]])
    df['_impact_text_norm'] = norm

    # --- 3) Helpers to build precise, word-bounded patterns ---
    # Optional connector between tokens: space or any dash, or nothing (to catch “startup” vs “start-up”)
    HX = r'(?:\s|[-–—])?'   # zero or one connector

    def wb(p: str) -> str:
        """
        Wrap p with 'letter' boundaries to avoid partial-word matches.
        Using A–Z bounds (ASCII letters) is appropriate for this English vocabulary.
        """
        return rf'(?<![A-Za-z]){p}(?![A-Za-z])'

    # --- 4) Define pattern lists per semantic group (phrases are case-insensitive) ---
    patterns = {
        # Charity / third sector
        'charity': [
            wb(r'charit(?:y|ies)'),
            wb(rf'non{HX}?profit(?:{HX}organi[sz]ation(?:s)?)?'),        # non-profit (organization/organisation)(s)
            wb(r'NGOs?'),
            wb(rf'non{HX}?governmental{HX}organi[sz]ation(?:s)?'),
            wb(rf'voluntary{HX}organi[sz]ation(?:s)?'),
            wb(rf'philanthropic{HX}organi[sz]ation(?:s)?'),
            wb(rf'third{HX}sector{HX}organi[sz]ation(?:s)?'),
            wb(rf'charitable{HX}trusts?'),
            wb(rf'charitable{HX}foundations?'),
            wb(rf'social{HX}enterprises?'),
        ],

        # Startups / spinouts / spinoffs
        'startup': [
            wb(rf'start{HX}?ups?'),          # startup, start-up, start up, startups, etc.
            wb(rf'spin{HX}?outs?'),          # spinout / spin-out(s)
            wb(rf'spin{HX}?offs?'),          # spinoff / spin-off(s)
        ],

        # Patents
        'patent': [
            wb(r'patent(?:s|ed|able|ing)?'), # patent, patents, patented, patentable, patenting
        ],

        # Museums / exhibitions / galleries
        'museum': [
            wb(r'museums?'),
            wb(rf'exhibition{HX}?s?'),
            wb(r'galler(?:y|ies)'),
        ],

        # NHS
        'nhs': [
            wb(r'NHS'),
            wb(rf'National{HX}Health{HX}Service'),
        ],

        # Drug / clinical / pharmaceutical trials and development/discovery
        'drug_trial': [
            wb(rf'(?:drug|pharmaceutical|clinical|medical){HX}trial(?:s)?'),
            wb(rf'(?:drug|pharmaceutical|therapeutic|medicine){HX}(?:development|discovery)'),
            wb(rf'new{HX}drug'),
        ],

        # Schools
        'school': [
            wb(r'schools?'),
        ],

        # Legislation / reform
        'legislation': [
            wb(r'legislations?'),
            wb(rf'legislative{HX}reform'),
            wb(rf'law{HX}(?:reform|change)'),
            wb(rf'legal{HX}reform'),
        ],

        # Heritage bodies
        'heritage': [
            wb(rf'National{HX}Trust'),
            wb(rf'English{HX}Heritage'),
            wb(rf'Historic{HX}England'),
            wb(r'UNESCO'),
        ],

        # Manufacturing and Software
        'manufacturing': [ wb(r'manufacturing') ],
        'software':      [ wb(r'software') ],
    }

    # --- 5) Compile combined regex per group (case-insensitive) ---
    compiled = {
        g: re.compile('|'.join(patts), flags=re.IGNORECASE)
        for g, patts in patterns.items()
    }

    # --- 6) Add indicator columns (booleans) ---
    for g, rx in compiled.items():
        df[f'ind_{g}'] = df['_impact_text_norm'].str.contains(rx, na=False)
    print('Merged in thematic indicators')
    return df.drop(columns=['_impact_text_norm'], inplace=True)


if __name__ == "__main__":
    # Project + data paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    project_root = Path.cwd()
    data_path = project_root / 'data'

    (raw_path, edit_path, sup_path,
     manual_path, final_path, topic_path,
     dim_path, openalex_path,
     ics_staff_rows_path,
     ics_grants_path) = get_paths(data_path)

    csv_out = [arg for arg in sys.argv if '.csv' in arg]
    if csv_out:
        output_path = csv_out[0]
        print(f"Will write final dataset to provided path: {output_path}")
        write = True
    else:
        output_path = final_path / 'enhanced_ref_data.zip'
        if not (final_path / 'enhanced_ref_data.zip').exists():
            print(f"Will write final dataset to default path: {output_path}")
            write = True
        else:
            if ('-f' in sys.argv):
                print(f"WARNING: Will force overwrite dataset at default path: {output_path}")
                write = True
            else:
                print("WARNING: Not overwriting final dataset as file already exists.\n"
                      "Use -f to force overwrite or provide a custom path.\n"
                      "Only intermittent files will be generated and saved.")
                write = False

    # Ensure raw artifacts exist (download if missing)
    if not os.path.exists(raw_path):
        os.makedirs(raw_path, exist_ok=True)
    if not (raw_path / 'raw_ref_environment_data.xlsx').exists():
        get_environmental_data(raw_path)
    get_all_results(raw_path)
    if not ((raw_path / 'raw_ref_ics_tags_data.xlsx').exists() and
            (raw_path / 'raw_ref_ics_data.xlsx').exists()):
        get_impact_data(raw_path)
    if not (raw_path / 'raw_ref_outputs_data.xlsx').exists():
        get_output_data(raw_path)

    clean_dep_level(raw_path, edit_path)
    df = clean_ics_level(raw_path, edit_path)
    df = load_dept_vars(df, edit_path)
    df = add_postcodes(df, sup_path)
    df = add_url(df)
    df = load_country(df, manual_path)
    df = load_funder(df, manual_path)
    df = make_and_load_tags(df, raw_path, edit_path)

    if '-bq' in sys.argv:
        print("Generating new Dimensions data... This will take some time.")
        my_project_id = return_dim_id(project_root / 'assets', 'dimensions_project_id.txt')
        if not ((dim_path / 'doi_returns_dimensions.xlsx').exists() and
                (dim_path / 'isbns_returns_dimensions.xlsx').exists() and
                (dim_path / 'title_returns_dimensions.xlsx').exists()):
            get_dimensions_data(manual_path, dim_path, my_project_id)
        else:
            if '-bqf' in sys.argv:
                print('Getting new Dimensions data')
                get_dimensions_data(manual_path, dim_path, my_project_id)
            else:
                print("Dimensions data already found, skipping collection "
                      "(use -bqf to force new collection). "
                      "Just generating paper level data from the dimensions data.")
        make_paper_level(dim_path)

    df = load_scientometric_data(df, dim_path, openalex_path)

    if '-top' in sys.argv:
        print("Generating new topic model... This will take some time.")
        bert_script_path = project_root / 'topic_modelling' / 'bert.py'
        run_args = [edit_path / 'clean_ref_ics_data.xlsx', topic_path]
        run_command = ["python3", str(bert_script_path), *map(str, run_args)]
        subprocess.run(run_command)

        print("Reducing topic model... This will take some time.")
        reduce_script_path = project_root / 'topic_modelling' / 'bert_reduce.py'
        reduce_args = [topic_path, 'nn3_threshold0.01']
        reduce_command = ["python3", str(reduce_script_path), *map(str, reduce_args)]
        subprocess.run(reduce_command)

    df = load_topic_data(df, manual_path, topic_path)

    if '-tm' in sys.argv:
        print("Generating new text-mining data... This will take some time.")
        section_columns = [
            "1. Summary of the impact",
            "2. Underpinning research",
            "3. References to the research",
            "4. Details of the impact",
            "5. Sources to corroborate the impact"
        ]
        gen_readability_scores(df, edit_path, section_columns)
        gen_pos_features(df, edit_path, section_columns)
        gen_sentiment_scores(df, edit_path, section_columns)

    df = get_readability_scores(df, edit_path)
    df = get_pos_features(df, edit_path)
    df = get_sentiment_scores(df, edit_path)

    df = get_ics_staff_rows(df, ics_staff_rows_path)
    df = get_ics_grants(df, ics_grants_path)
    df = get_university_class(df, manual_path)
    df = get_panel_and_UoA_names(df)
    df = get_thematic_indicators(df)

    if write:
        df.to_csv(output_path,
                  index=False,
                  compression=dict(method='zip',
                                   archive_name='enhanced_ref_data.csv'))
