"""Build the public static site from the dashboard page and the latest weekly data.

Usage: python3 -I build_site.py <work_dir> <season> <site_dir> [site_url]
Writes <site_dir>/index.html, data/*.json, d3.min.js (+license), og.png, .nojekyll.
"""
import sys, os, json, shutil, csv

work, season, site = os.path.abspath(sys.argv[1]), int(sys.argv[2]), os.path.abspath(sys.argv[3])
url = sys.argv[4] if len(sys.argv) > 4 else 'https://thomasconner.github.io/nfl-prediction-model/'
src = f'{work}/site_src'
os.makedirs(f'{site}/data', exist_ok=True)
SOURCES = ['meta', 'weekly', 'games', 'team_games', 'cards', 'accuracy', 'verdicts']
for n in SOURCES[:-1]:
    shutil.copyfile(f'{work}/out/dash_{season}/{n}.json', f'{site}/data/{n}.json')
rows = []
for r in csv.DictReader(open(f'{work}/out/verdict_backtest.csv')):
    rows.append({k: (v if k == 'verdict' else float(v)) for k, v in r.items()})
json.dump(rows, open(f'{site}/data/verdicts.json', 'w'), separators=(',', ':'))
page = open(f'{src}/page.html').read().replace('<meta name="dash-one-writer">\n', '')
shell = open(f'{src}/shell.html').read()
html = shell.replace('__SOURCES__', json.dumps(SOURCES)).replace('__SITE_URL__', url).replace('__PAGE__', page)
open(f'{site}/index.html', 'w').write(html)
for f in ('d3.min.js', 'd3.LICENSE.txt', 'og.png'):
    if os.path.exists(f'{src}/{f}'):
        shutil.copyfile(f'{src}/{f}', f'{site}/{f}')
open(f'{site}/.nojekyll', 'w').write('')
print('site built', site, sorted(os.listdir(site)), sorted(os.listdir(f'{site}/data')))
