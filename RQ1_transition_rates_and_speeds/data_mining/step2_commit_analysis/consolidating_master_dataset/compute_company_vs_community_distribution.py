#!/usr/bin/env python3
import sys
import csv
from collections import defaultdict, Counter


def classify_owner(owner: str) -> str:
    o = (owner or '').strip().lower()
    if not o:
        return 'unknown'

    # Heuristic lists; conservative bias toward community if unknown
    company_owners = {
        'google', 'googleapis', 'google-research', 'chromium', 'android', 'tensorflow', 'bazelbuild',
        'microsoft', 'azure', 'dotnet', 'microsoftgraph',
        'meta', 'facebook', 'fbsamples',
        'uber', 'airbnb', 'square', 'paypal', 'stripe', 'shopify', 'netflix', 'linkedin', 'oracle', 'ibm',
        'adobe', 'intel', 'nvidia', 'bytedance', 'alibaba', 'baidu', 'tencent', 'salesforce', 'heroku',
        'atlassian', 'elastic', 'databricks', 'confluentinc', 'redhat', 'aws', 'awslabs', 'amzn',
        'docker', 'hashicorp', 'github', 'gitlab', 'canonical', 'jetbrains', 'yandex', 'spotify',
        'twitter', 'x', 'pinterest', 'dropbox', 'cloudflare', 'digitalocean', 'cisco', 'juniper',
        'samsung', 'apple', 'arm', 'sony', 'nokia', 'wechat', 'meituan', 'huawei', 'bytedance', 'antfin',
    }

    foundation_owners = {
        'apache', 'eclipse', 'openstack', 'mozilla', 'linuxfoundation', 'cncf', 'kubernetes', 'lfai',
        'theupdateframework', 'cdfoundation', 'openjs-foundation', 'openjs', 'opentelemetry', 'opentracing',
        'openstack', 'cloudfoundry', 'joomla', 'drupal', 'gnome', 'kde', 'qt', 'opendaylight',
    }

    if 'foundation' in o or o.endswith('-foundation') or o.endswith('foundation'):
        return 'community-led'
    if o in foundation_owners:
        return 'community-led'
    if o in company_owners:
        return 'company-backed'
    return 'community-led'


def main():
    if len(sys.argv) < 2:
        print('Usage: python compute_company_vs_community_distribution.py <master_commits_dataset.csv>')
        sys.exit(1)
    path = sys.argv[1]

    # Collect unique projects and their type
    projects = {}
    owners = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pn = row.get('project_name', '').strip()
            pt = row.get('project_type', '').strip()
            if not pn:
                continue
            if pn not in projects:
                projects[pn] = pt
                owners[pn] = pn.split('/')[0].strip() if '/' in pn else pn

    # Classify
    counts_overall = Counter()
    counts_by_type = defaultdict(Counter)
    owners_by_class = defaultdict(set)

    for pn, pt in projects.items():
        owner = owners[pn]
        cl = classify_owner(owner)
        counts_overall[cl] += 1
        if pt:
            counts_by_type[pt][cl] += 1
        owners_by_class[cl].add(owner)

    total = sum(counts_overall.values())
    print('Company-backed vs Community-led distribution (by unique project):')
    for cl in ['company-backed', 'community-led', 'unknown']:
        if counts_overall[cl]:
            pct = 100.0 * counts_overall[cl] / total if total else 0.0
            print(f"  {cl}: {counts_overall[cl]} / {total} ({pct:.1f}%)")

    print('\nBreakdown by project_type:')
    for pt in sorted(counts_by_type.keys()):
        subtotal = sum(counts_by_type[pt].values())
        print(f"  {pt} (n={subtotal}): ", end='')
        parts = []
        for cl in ['company-backed', 'community-led', 'unknown']:
            c = counts_by_type[pt][cl]
            if c:
                parts.append(f"{cl} {c} ({100.0*c/subtotal:.1f}%)")
        print('; '.join(parts))

    # Show top owners to sanity-check classification
    print('\nSample owners by class (first 20 each):')
    for cl in ['company-backed', 'community-led', 'unknown']:
        if owners_by_class[cl]:
            sample = sorted(owners_by_class[cl])[:20]
            print(f"  {cl}: {', '.join(sample)}")


if __name__ == '__main__':
    main()

