# Utilities related to benthic attribute info from the MERMAID API.

from collections import defaultdict
import csv
import dataclasses
import json
import operator
import urllib.request


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
        return self.by_id[ba_id]['name']

    def name_to_id(self, ba_name):
        return self.by_name[ba_name]['id']

    def bagf_id_to_name(self, bagf_id, gf_library):
        if '::' in bagf_id:
            ba_id, gf_id = bagf_id.split('::')
            ba_name = self.by_id[ba_id]['name']
            return ba_name + '::' + gf_library.by_id[gf_id]
        else:
            return self.by_id[bagf_id]['name']

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


@dataclasses.dataclass
class LabelMappingEntry:
    # Since this is a Python dataclass, there's an automatically generated
    # __init__() method which sets the fields below.

    # MERMAID UUIDs
    benthic_attribute_id: str
    growth_form_id: str | None
    # Human-readable names from the provider site (what we're mapping from)
    provider_label: str


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
        if self._mapping is None:
            self._load_mapping()

        return cn_label_id in self._mapping

    def __getitem__(self, cn_label_id):
        if self._mapping is None:
            self._load_mapping()

        try:
            return self._mapping[cn_label_id]
        except KeyError as e:
            raise KeyError(f"{e} - Make sure you're passing the CoralNet label ID (not name), as a string (not int).")

    def add_bagf_in_duckdb(
        self,
        cn_label_ids: list[str],
        duck_conn: 'DuckDBPyConnection',
        duck_table_name: str,
        cn_label_id_column_name: str = 'label_id',
        benthic_attribute_id_column_name: str = 'benthic_attribute_id',
        growth_form_id_column_name: str = 'growth_form_id',
    ):
        """
        Create columns for benthic attribute and growth form, based on
        the CoralNet label ID column, in the given DuckDB database.

        To make the operation more lightweight, only consider the
        label IDs given in `cn_label_ids`. These should be the only
        label IDs that occur in the data.

        Any label IDs not in the API's mapping map to BA=NULL, GF=NULL.
        """
        mapping_entries = [
            (
                cn_id,
                self[cn_id].benthic_attribute_id,
                self[cn_id].growth_form_id,
            )
            for cn_id in cn_label_ids
            if cn_id in self
        ]
        # Handle the API itself giving a BA without a GF.
        # '<growth-form-uuid>' if present, NULL if not present.
        mapping_str = ', '.join(
            f"('{cn}', '{ba}', {'\''+gf+'\'' if gf else 'NULL'})"
            for cn, ba, gf in mapping_entries)

        # Create a DuckDB table for the label mapping.
        duck_conn.execute(
            f"CREATE TABLE label_mapping"
            f" ({cn_label_id_column_name} VARCHAR,"
            f"  {benthic_attribute_id_column_name} VARCHAR,"
            f"  {growth_form_id_column_name} VARCHAR)")
        duck_conn.execute(
            f"INSERT INTO label_mapping VALUES"
            f" {mapping_str}"
        )

        # Use table joining to apply the label mapping.
        duck_conn.execute(
            f"CREATE OR REPLACE TABLE {duck_table_name} AS"
            f" SELECT *"
            f" FROM {duck_table_name}"
            f" JOIN label_mapping"
            f" ON {duck_table_name}.{cn_label_id_column_name}"
            f"  = label_mapping.{cn_label_id_column_name}"
        )

    def _load_mapping(self):
        endpoint_response = urllib.request.urlopen(self._endpoint)
        response_json = json.loads(endpoint_response.read())
        api_mapping = response_json['results']

        while response_json['next']:
            endpoint_response = urllib.request.urlopen(response_json['next'])
            response_json = json.loads(endpoint_response.read())
            api_mapping.extend(response_json['results'])

        self._mapping = {
            entry['provider_id']: LabelMappingEntry(
                benthic_attribute_id=entry['benthic_attribute_id'],
                growth_form_id=entry['growth_form_id'],
                provider_label=entry['provider_label'],
            )
            for entry in api_mapping
        }


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
