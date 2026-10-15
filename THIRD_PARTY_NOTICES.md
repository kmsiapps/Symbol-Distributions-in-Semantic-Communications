# Third-party notices and licensing scope

The root [LICENSE](LICENSE) covers only original project material Hanju Yoo is authorized to license. It does not grant rights to the components below, and it does not establish redistribution rights for material whose upstream terms are unclear. Review the listed upstream terms and obtain any needed permission before redistributing those components.

## ELIC implementation

`models/ELIC/` identifies its source as [VincentChandelier/ELiC-ReImplemetation](https://github.com/VincentChandelier/ELiC-ReImplemetation). The upstream repository currently has no repository-level license file. Files under `models/ELIC/` without a clear file-level license are not covered by this repository's MIT license; permission to redistribute them has not been established here.

Some files under `models/ELIC/ELICUtilis/` carry InterDigital copyright headers and Apache License 2.0 notices. Those file-level notices remain applicable. A copy of Apache License 2.0 is included at [THIRD_PARTY_LICENSES/Apache-2.0.txt](THIRD_PARTY_LICENSES/Apache-2.0.txt). Check each file's header for exact attribution and scope.

## M2T implementation

`models/M2T/` identifies its source as [wsxtyrdd/Masked-Transformer-For-Image-Compression](https://github.com/wsxtyrdd/Masked-Transformer-For-Image-Compression). The upstream repository currently has no repository-level license file. Although a local file header refers to MIT, the upstream repository does not provide a matching license grant that could be verified. This code is not covered by the root MIT license; redistribution rights need confirmation.

## TCM implementation

`models/TCM/` identifies [jmliu206/LIC_TCM](https://github.com/jmliu206/LIC_TCM) as its source. That upstream repository provides an MIT license with copyright notice `Copyright (c) 2023 jmliu206`. A copy is included at [THIRD_PARTY_LICENSES/LIC_TCM-MIT.txt](THIRD_PARTY_LICENSES/LIC_TCM-MIT.txt). Preserve that notice when redistributing the applicable TCM code.

## Data, weights, images, and dependencies

The root MIT license does not cover datasets or their contents, pretrained model weights, test images, or other assets. In particular, confirm the provenance and redistribution terms for files under `results/raw_data/` before making them publicly downloadable. Dependencies such as CompressAI retain their own licenses; consult the relevant dependency's distribution and license files.

## Research paper

The associated paper is published by IEEE. This code license does not license the paper or its figures; follow the rights and terms published with the article.
