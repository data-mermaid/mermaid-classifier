# Utilities related to benthic attribute info from the MERMAID API.

from collections import defaultdict
import csv
import json
import operator
import urllib.request


class BenthicAttrHierarchy:

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


class GrowthForms:

    def __init__(self):
        download_response = urllib.request.urlopen(
            'https://api.datamermaid.org/v1/choices/')
        response_json = json.loads(download_response.read())
        for item in response_json:
            if item['name'] == 'growthforms':
                data = item['data']
                break
        self.lookup = {gf['id']: gf['name'] for gf in data}


def output_ba_csvs():

    hierarchy = BenthicAttrHierarchy()

    raw_path = 'benthic_attributes_raw.csv'
    fieldnames = ['id', 'name', 'parent']
    print(f"Writing to: {raw_path}")
    with open(raw_path, 'w', newline='', encoding='utf-8') as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=fieldnames)
        writer.writeheader()
        for result in hierarchy.raw_results:
            writer.writerow(
                {k: v for k, v in result.items() if k in fieldnames})

    results_ordered_by_parent = hierarchy.get_descendants(None)
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
                parent_name = hierarchy.id_to_name(result['parent'])
            writer.writerow(dict(
                name=result['name'],
                parent_name=parent_name,
            ))


if __name__ == '__main__':

    output_ba_csvs()
