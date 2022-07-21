<a href="https://github.com/jfcrenshaw/pzflow-paper/actions/workflows/build.yml">
<img src="https://github.com/jfcrenshaw/pzflow-paper/actions/workflows/build.yml/badge.svg" alt="Article status"/>
</a>
<a href="https://github.com/jfcrenshaw/pzflow-paper/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a>
<a href="https://github.com/jfcrenshaw/pzflow-paper/raw/main-pdf/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>

Journal article for [PZFlow](https://jfcrenshaw.github.io/pzflow/) created using the [showyourwork](https://github.com/showyourwork/showyourwork) workflow.

To Do:

- clean up the train ensemble script
- fix the artifact in the pz posteriors at z=3 (probably doable by increasing the redshift range?)
- when combining the loss dictionaries for the ensemble training, wrap the losses in float() so that they're not jax arrays and can be unpickled
- add posterior estimation with missing u bands and add the missing u band variety to all of the metrics
