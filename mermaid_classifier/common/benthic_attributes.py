# Utilities related to benthic attribute info from the MERMAID API.

from collections import defaultdict
import csv
import dataclasses
import json
import operator
import urllib.request

import pandas as pd


# MERMAID API uses :: as the BA-GF separator.
BAGF_SEP = '::'


def combine_ba_gf(
    benthic_attribute: str, growth_form: str,
) -> str:
    """
    Combine string representations of a benthic attribute and a growth form
    into a single representation for the BA-GF combo.
    If there's no GF (empty string), the result has the BA + separator.
    Not just the BA.
    """
    return BAGF_SEP.join([benthic_attribute, growth_form])


def split_ba_gf(bagf: str) -> tuple[str, str]:
    """
    Split a string representation of a BA-GF combo into individual strings
    for benthic attribute and growth form.
    If there's no GF, the input should have the BA + separator. Not just
    the BA.
    """
    try:
        benthic_attribute, growth_form = bagf.split(BAGF_SEP)
    except ValueError:
        raise ValueError(
            f"'{bagf}' is not a valid BA-GF combo string."
            f" The separator {BAGF_SEP} should appear exactly once.")

    if benthic_attribute == '':
        raise ValueError(
            f"'{bagf}' is not a valid BA-GF combo string."
            f" There should be characters to the left of the"
            f" separator {BAGF_SEP}.")

    return benthic_attribute, growth_form


class BenthicAttributeLibrary:
    """
    Information about all MERMAID benthic attributes, organized into
    various lookups.
    This is intended to be a singleton class.
    """
    def __init__(self):
        download_response = urllib.request.urlopen(
            'https://api.datamermaid.org/v1/benthicattributes/?limit=5000')
        response_json = json.loads(download_response.read())
        self.raw_results = response_json['results']

        self.by_id = dict()
        self.by_name = dict()
        self.by_parent = defaultdict(list)

        for result in self.raw_results:
            self.by_id[result['id']] = result
            self.by_name[result['name']] = result
            self.by_parent[result['parent']].append(result)

    def id_to_name(self, ba_id):
        if ba_id == '':
            return ''
        return self.by_id[ba_id]['name']

    def name_to_id(self, ba_name):
        if ba_name == '':
            return ''
        return self.by_name[ba_name]['id']

    def bagf_id_to_name(self, bagf_id, gf_library):
        ba_id, gf_id = split_ba_gf(bagf_id)
        ba_name = self.by_id[ba_id]['name']
        if gf_id == '':
            # Unlike IDs where we represent BA-only as <BA><separator>,
            # for readable names BA-only will have no separator.
            return ba_name
        else:
            return BAGF_SEP.join([ba_name, gf_library.by_id[gf_id]])

    def get_ancestor_ids(self, ba_id):
        """
        Get ancestor IDs, ordered earliest (closest to root) first.
        """
        parent = self.by_id[ba_id]['parent']
        if parent is None:
            return []
        return self.get_ancestor_ids(parent) + [parent]

    def get_descendants(self, ba_id):
        """
        Get descendants as a list that's ordered by parent, with parents
        being in a depth-first-search order.
        """
        if ba_id not in self.by_parent:
            # No children
            return []

        children = self.by_parent[ba_id]
        children_ordered_by_name = sorted(
            children, key=operator.itemgetter('name'))

        children_results = []
        for child in children_ordered_by_name:
            children_results.extend(
                self.get_descendants(child['id']))
        return children_ordered_by_name + children_results


class GrowthFormLibrary:
    """
    Information about MERMAID growth forms, primarily an id-to-name lookup.
    This is intended to be a singleton class.
    """
    def __init__(self):
        download_response = urllib.request.urlopen(
            'https://api.datamermaid.org/v1/choices/')
        response_json = json.loads(download_response.read())
        for item in response_json:
            if item['name'] == 'growthforms':
                data = item['data']
                break
        self.by_id = {gf['id']: gf['name'] for gf in data}

    def id_to_name(self, gf_id):
        if gf_id == '':
            return ''
        return self.by_id[gf_id]


@dataclasses.dataclass
class LabelMappingEntry:
    # Since this is a Python dataclass, there's an automatically generated
    # __init__() method which sets the fields below.

    # We map from the provider's ID/label to the benthic attribute
    # and growth form.
    # Not all these fields are strictly necessary for the task of mapping,
    # but they're useful to include for MLflow artifacts.
    #
    # Note that the order the fields appear here also determines the order
    # they appear in the MLflow artifact, assuming these dataclass instances
    # are passed directly into a pd.DataFrame() which serves as the artifact.
    #
    # Lack of a growth form is indicated by an empty string for those fields.
    provider_label: str
    benthic_attribute_name: str
    growth_form_name: str
    provider_id: str
    benthic_attribute_id: str
    growth_form_id: str


class CoralNetMermaidMapping:
    """
    Mapping from CoralNet label IDs to:
    - MERMAID benthic attribute IDs (benthic_attribute_id)
    - MERMAID growth form IDs (growth_form_id)
    - CoralNet label names (provider_label)

    Loaded from the MERMAID API. Lazy-loaded (loaded only when a lookup
    is actually made) and cached (only loaded from API once).
    """

    def __init__(
        self,
        mapping_endpoint='https://api.datamermaid.org/v1/classification/labelmappings/?provider=CoralNet',
    ):
        self._mapping = None
        self._endpoint = mapping_endpoint

    def __contains__(self, cn_label_id):
        return cn_label_id in self.mapping

    def __getitem__(self, cn_label_id):
        try:
            return self.mapping[cn_label_id]
        except KeyError as e:
            raise KeyError(f"{e} - Make sure you're passing the CoralNet label ID (not name), as a string (not int).")

    def get_dataframe(self):
        return pd.DataFrame(self.mapping.values())

    @property
    def mapping(self):
        if self._mapping is None:
            self._load_mapping()
        return self._mapping

    def _load_mapping(self):
        api_mapping = self._download_mapping()

        self._mapping = {
            entry['provider_id']: LabelMappingEntry(
                benthic_attribute_id=entry['benthic_attribute_id'],
                # Use '' as the empty value for GFs, not null/None.
                growth_form_id=entry['growth_form_id'] or '',
                benthic_attribute_name=entry['benthic_attribute_name'],
                growth_form_name=entry['growth_form_name'] or '',
                provider_id=entry['provider_id'],
                provider_label=entry['provider_label'],
            )
            for entry in api_mapping
        }

    def _download_mapping(self):
        endpoint_response = urllib.request.urlopen(self._endpoint)
        response_json = json.loads(endpoint_response.read())
        api_mapping = response_json['results']

        while response_json['next']:
            endpoint_response = urllib.request.urlopen(response_json['next'])
            response_json = json.loads(endpoint_response.read())
            api_mapping.extend(response_json['results'])

        return api_mapping


def output_ba_csvs():
    """
    Output CSV files of all the benthic attributes into the current directory.
    Not the most glamorous of visualizations, but can still be handy.
    """
    benthic_attrs = BenthicAttributeLibrary()

    raw_path = 'benthic_attributes_raw.csv'
    fieldnames = ['id', 'name', 'parent']
    print(f"Writing to: {raw_path}")
    with open(raw_path, 'w', newline='', encoding='utf-8') as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=fieldnames)
        writer.writeheader()
        for result in benthic_attrs.raw_results:
            writer.writerow(
                {k: v for k, v in result.items() if k in fieldnames})

    results_ordered_by_parent = benthic_attrs.get_descendants(None)
    ordered_path = 'benthic_attributes_ordered_by_parent.csv'
    fieldnames = ['name', 'parent_name']
    print(f"Writing to: {ordered_path}")
    with open(ordered_path, 'w', newline='', encoding='utf-8') as ordered_f:
        writer = csv.DictWriter(ordered_f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results_ordered_by_parent:
            if result['parent'] is None:
                parent_name = None
            else:
                parent_name = benthic_attrs.id_to_name(result['parent'])
            writer.writerow(dict(
                name=result['name'],
                parent_name=parent_name,
            ))
