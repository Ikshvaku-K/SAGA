# Third-party notices

## BDD100K dataset

The sample images in `app/static/samples/` and the metrics in `results/` and `remote_runs/` were produced from the Berkeley DeepDrive BDD100K dataset:

> Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved.
> Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
>
> IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

The dataset itself is not included in this repository; download it from https://bdd-data.berkeley.edu/. Commercial use requires a separate license from UC Berkeley.

## Pretrained model

SegFormer-B2 weights pre-trained on Cityscapes (`nvidia/segformer-b2-finetuned-cityscapes-1024-1024`) are downloaded from Hugging Face at runtime and are subject to NVIDIA's license for that model.

## Map tiles

Map tiles © OpenStreetMap contributors (ODbL), loaded at runtime from tile.openstreetmap.org. Map rendering by Leaflet (BSD-2-Clause), loaded from unpkg.

## Voice

Spoken summaries are generated at runtime with the ElevenLabs API under the deployer's own ElevenLabs account and terms.
