"""Chile market: Spanish resume without address, 100% remote filter, tracker columns, folders, GetOnBoard.

Canada's behavior (English/French, address, file names, folders) is checked here too: it must not change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openpyxl import load_workbook

from job_assistant.cv import postular
from job_assistant.cv.content import Bullet, JobBullets, ResumeContent
from job_assistant.cv.jd import JobDescription
from job_assistant.cv.master import CVMaster, load_master
from job_assistant.cv.render import build_html
from job_assistant.cv.roles import load_roles
from job_assistant.discovery import chile as getonboard
from job_assistant.tailoring.chile import (
    NOT_STATED, check_fully_remote, extract_details, is_chile, spanish_roles,
)
from job_assistant.tracker.template import create_tracker_template
from job_assistant.tracker.workbook import write_header
from tests.test_cv import (  # reuse the CV test doubles
    EXAMPLES, PROFILE, TODAY, FakeRequirements, FakeReviewer, FakeWriter, fake_render, review,
)

SPANISH_OVERLAY = """
[contact_es]
phone = "(647) 830-6418"

[meta]
banned_terms_es = ["fluido"]

[default_title]
es = "Desarrollador Java Senior"

[[experience]]
id = "bank"
company_es = "Banco Ejemplo Canadá"
location_es = "Toronto, ON"
title_es = "Ingeniero de Software"
dates_es = "Enero 2020 – Actualidad"
stack_es = ["Java"]
bullets_es = ["Construí microservicios con Spring Boot para pagos."]

[[experience]]
id = "insurer"
location_es = "Santiago, Chile"
title_es = "Desarrollador"
dates_es = "Enero 2015 – Diciembre 2019"
bullets_es = ["Migré procesos batch de Oracle PL/SQL a servicios Java."]

[education]
es = ["Ingeniero Informático — Duoc UC — diciembre 2012"]

[certifications]
es = "AWS Cloud Practitioner (completada)"

[languages]
es = "Español, inglés, francés"

[skills]
tools_es = ["Java", "Spring Boot"]
methodologies_es = ["Agile"]

[summary_facts]
es = "Desarrollador Java con experiencia en microservicios."
"""

SPANISH_JD = ("Desarrollador Java Senior. Trabajo 100% remoto desde Chile. Contrato indefinido. "
              "Renta líquida $3.000.000 - $3.500.000 CLP. Debes residir en Chile. " * 4)
HYBRID_JD = "Desarrollador Java. Modalidad híbrida, 3 días a la semana en oficina en Santiago. " * 6


@pytest.fixture
def master_dir(tmp_path: Path) -> Path:
    (tmp_path / "cv_master.toml").write_bytes((EXAMPLES / "cv_master.example.toml").read_bytes())
    (tmp_path / "cv_master_es.toml").write_text(SPANISH_OVERLAY, encoding="utf-8")
    return tmp_path


@pytest.fixture
def master(master_dir: Path) -> CVMaster:
    m = load_master(master_dir, allow_example=True)
    # The example has no work authorization: give it one so the tests can see it disappear in Chile.
    return CVMaster(contact={**m.contact, "work_authorization": "Canadian Permanent Resident"}, raw=m.raw)


def spanish_content() -> ResumeContent:
    return ResumeContent(
        title="Desarrollador Java Senior", summary="Desarrollador Java con experiencia en microservicios.",
        competencies=["Java"],
        experience=[JobBullets(job_id="bank", bullets=[Bullet(lead="Construí microservicios", rest="con Spring Boot.")]),
                    JobBullets(job_id="insurer", bullets=[Bullet(lead="Migré procesos", rest="batch a Java.")])],
        tools=["Java"], methodologies=["Agile"],
    )


def english_content() -> ResumeContent:
    return ResumeContent(
        title="Senior Java Developer", summary="Java developer.", competencies=["Java"],
        experience=[JobBullets(job_id="bank", bullets=[Bullet(lead="Built services", rest="for payments.")]),
                    JobBullets(job_id="insurer", bullets=[Bullet(lead="Migrated jobs", rest="to Java.")])],
        tools=["Java"], methodologies=["Agile"],
    )


# ---- the Spanish overlay -----------------------------------------------------------------------


def test_spanish_overlay_only_adds_keys(master_dir: Path) -> None:
    with_es = load_master(master_dir, allow_example=True)
    (master_dir / "cv_master_es.toml").unlink()
    without_es = load_master(master_dir, allow_example=True)
    for lang in ("en", "fr"):
        if without_es.has_language(lang):
            assert [j.bullets for j in with_es.jobs(lang)] == [j.bullets for j in without_es.jobs(lang)]
        assert with_es.source_text("en") == without_es.source_text("en")
    assert without_es.has_language("es") is False
    assert with_es.has_language("es") is True


def test_spanish_company_override_and_banned_terms(master: CVMaster) -> None:
    jobs = {j.id: j for j in master.jobs("es")}
    assert jobs["bank"].company == "Banco Ejemplo Canadá"
    assert jobs["insurer"].company == master.jobs("en")[1].company  # no override: same employer name
    assert "fluido" in master.banned_terms("es") and "fluido" not in master.banned_terms("en")


# ---- the header: address per market --------------------------------------------------------------


def test_canada_keeps_address_phone_and_authorization(master: CVMaster) -> None:
    html = build_html(master, english_content(), "en")
    head = html.split("PROFESSIONAL SUMMARY")[0]
    assert "Toronto, ON, Canada" in head and "(000) 000-0000" in head and "Canadian Permanent Resident" in head


def test_chile_header_has_no_address_nor_authorization(master: CVMaster) -> None:
    html = build_html(master, spanish_content(), "es")
    head = html.split("RESUMEN PROFESIONAL")[0]
    assert "alex@example.com | (647) 830-6418 | linkedin.com/in/alex-sample" in head
    assert "(000) 000-0000" not in head and "Chile" not in head
    assert "Toronto" not in head and "Canadian" not in head and "Resident" not in head
    assert "EXPERIENCIA PROFESIONAL" in html and "Stack técnico" in html and "Banco Ejemplo Canadá" in html


# ---- remote filter and details --------------------------------------------------------------------


@pytest.mark.parametrize("text,ok", [
    ("Trabajo 100% remoto desde cualquier parte de Chile", True),
    ("Cargo remoto, equipo distribuido", True),
    ("Modalidad híbrida, 2 días presenciales", False),
    ("Presencial en Providencia", False),
    ("Remote or on-site in Santiago", False),
    ("Remoto. No es presencial ni híbrido.", True),
    ("Desarrollador Java en Santiago", False),
    ("Hybrid role, 100% remote option", True),
])
def test_fully_remote_filter(text: str, ok: bool) -> None:
    assert check_fully_remote(text).fully_remote is ok


def test_extract_details() -> None:
    d = extract_details("Contrato indefinido. Renta líquida $3.000.000 - $3.500.000. Debes residir en Chile.")
    assert (d.contract, d.requires_residence) == ("Indefinido", "Sí")
    assert d.salary_clp == "$3.000.000 - $3.500.000 CLP líquidos"
    d = extract_details("Contrato a honorarios. Desde cualquier lugar de LATAM. Sueldo USD 4000.")
    assert (d.contract, d.requires_residence, d.salary_clp) == ("Honorarios", "No", NOT_STATED)
    d = extract_details("Plazo fijo de 6 meses. Pago US$ 5.000.000.")
    assert (d.contract, d.salary_clp) == ("Plazo fijo", NOT_STATED)
    assert extract_details("Sin datos.") == extract_details("nada") and extract_details("x").contract == NOT_STATED


def test_is_chile_only_when_country_says_so() -> None:
    assert is_chile("Chile") and is_chile(" chile ") and not is_chile("") and not is_chile(None) and not is_chile("Canadá")


# ---- the tracker ---------------------------------------------------------------------------------


def test_new_columns_are_appended_at_the_end() -> None:
    fields = [c.field for c in postular.COLUMNS]
    assert fields[:17] == ["link", "company", "role", "location", "language", "notes", "description", "status",
                           "cv_file", "hr_score", "hr_notes", "processed", "fit", "approved", "source", "posted",
                           "master_role"]
    assert fields[17:] == ["country", "requires_residence", "contract", "salary_clp"]


def test_old_sheet_gets_the_columns_without_moving_anything(tmp_path: Path) -> None:
    path = create_tracker_template(tmp_path / "tracker.xlsx")
    wb = load_workbook(path)
    ws = wb.create_sheet("Postular", 0)
    old = postular.COLUMNS[:17]
    write_header(ws, [(c.field, c.header, c.width) for c in old])
    ws.cell(2, 1, "https://old")
    ws.cell(2, 8, "Aplicado")
    wb.save(path)
    assert postular.ensure_sheet(path) is False
    ws = load_workbook(path)["Postular"]
    assert [ws.cell(1, i).value for i in range(1, 18)] == [c.header for c in old]
    assert [ws.cell(1, i).value for i in range(18, 22)] == ["País", "Requiere residir en Chile", "Tipo de contrato", "Sueldo (CLP)"]
    assert ws.cell(2, 1).value == "https://old" and ws.cell(2, 8).value == "Aplicado"
    assert [r["country"] for r in postular.table(path).rows()] == [None]  # an old row has no country: Canada


@pytest.fixture
def sheet_path(tmp_path: Path) -> Path:
    path = create_tracker_template(tmp_path / "tracker.xlsx")
    postular.ensure_sheet(path)
    return path


def run_row(sheet_path: Path, master: CVMaster, tmp_path: Path, row: dict[str, Any], jd: str,
            draft: ResumeContent) -> postular.RowOutcome:
    return postular.process_row(
        row, path=sheet_path, master=master, profile=PROFILE, requirements=FakeRequirements(False),  # type: ignore[arg-type]
        writer=FakeWriter([draft]), reviewer=FakeReviewer([review(9)]),  # type: ignore[arg-type]
        output_dir=tmp_path / "out", fetch=lambda url: JobDescription(text=jd, title="Desarrollador Java"),
        renderer=fake_render, today=TODAY,
    )


def test_chile_row_goes_to_chile_folder_in_spanish(sheet_path: Path, master: CVMaster, tmp_path: Path) -> None:
    postular.table(sheet_path).append({"link": "https://getonbrd.com/jobs/1", "country": "Chile", "company": "Acme CL"})
    [row] = postular.pending_rows(sheet_path)
    out = run_row(sheet_path, master, tmp_path, row, SPANISH_JD, spanish_content())
    assert out.status == postular.STATUS_READY and out.cv_path is not None
    assert out.cv_path.parent.parent.name == "Chile" and out.cv_path.parent.parent.parent.name == "Resumenes 10-1"
    assert out.cv_path.name == "Alex_Sample_CV_ES_Acme-Cl_Desarrollador-Java.pdf"
    html = out.cv_path.read_text(encoding="utf-8")
    assert "RESUMEN PROFESIONAL" in html and "Toronto, ON, Canada" not in html.split("RESUMEN PROFESIONAL")[0]
    [saved] = postular.table(sheet_path).rows()
    assert saved["language"] == "es" and saved["country"] == "Chile"
    assert (saved["requires_residence"], saved["contract"]) == ("Sí", "Indefinido")
    assert saved["salary_clp"] == "$3.000.000 - $3.500.000 CLP líquidos"
    assert postular.find_cv_folder(tmp_path / "out", out.cv_path.name) == out.cv_path.parent


def test_hybrid_chile_row_is_discarded_without_a_resume(sheet_path: Path, master: CVMaster, tmp_path: Path) -> None:
    postular.table(sheet_path).append({"link": "https://x/2", "country": "Chile"})
    [row] = postular.pending_rows(sheet_path)
    out = run_row(sheet_path, master, tmp_path, row, HYBRID_JD, spanish_content())
    assert out.status == "Descartado" and out.cv_path is None
    assert not (tmp_path / "out").exists()
    [saved] = postular.table(sheet_path).rows()
    assert saved["status"] == "Descartado" and "hibrid" in str(saved["hr_notes"])
    assert postular.pending_rows(sheet_path) == []  # not picked up again


def test_forzar_generates_a_hybrid_chile_row(sheet_path: Path, master: CVMaster, tmp_path: Path) -> None:
    postular.table(sheet_path).append({"link": "https://x/3", "country": "Chile", "status": "Forzar"})
    [row] = postular.pending_rows(sheet_path)
    out = run_row(sheet_path, master, tmp_path, row, HYBRID_JD, spanish_content())
    assert out.status == postular.STATUS_READY and "Chile" in out.cv_path.parts  # type: ignore[union-attr]


def test_canada_row_is_unchanged(sheet_path: Path, master: CVMaster, tmp_path: Path) -> None:
    postular.table(sheet_path).append({"link": "https://job-boards.greenhouse.io/acme/jobs/1"})
    [row] = postular.pending_rows(sheet_path)
    out = run_row(sheet_path, master, tmp_path, row, "Senior Java Developer. We need Java. " * 10, english_content())
    assert out.cv_path is not None and "Chile" not in out.cv_path.parts
    assert out.cv_path.name == "Alex_Sample_Resume_Acme_Java-Dev.pdf" or out.cv_path.name.startswith("Alex_Sample_Resume_")
    head = out.cv_path.read_text(encoding="utf-8").split("PROFESSIONAL SUMMARY")[0]
    assert "Toronto, ON, Canada" in head and "Canadian Permanent Resident" in head
    [saved] = postular.table(sheet_path).rows()
    assert saved["language"] == "en" and saved["country"] is None and saved["salary_clp"] is None


def test_chile_without_spanish_master_falls_back_to_english(sheet_path: Path, master_dir: Path, tmp_path: Path) -> None:
    (master_dir / "cv_master_es.toml").unlink()
    plain = load_master(master_dir, allow_example=True)
    postular.table(sheet_path).append({"link": "https://x/4", "country": "Chile"})
    [row] = postular.pending_rows(sheet_path)
    out = run_row(sheet_path, plain, tmp_path, row, SPANISH_JD, english_content())
    assert "_CV_ES_" not in out.cv_path.name and "Resume" in out.cv_path.name  # type: ignore[union-attr]
    assert "cv_master_es.toml" in out.detail


def test_spanish_titles_pick_the_right_role() -> None:
    from job_assistant.cv.roles import match_role

    roles = spanish_roles(load_roles(Path(__file__).resolve().parent.parent / "data")) if (
        Path(__file__).resolve().parent.parent / "data" / "roles.toml").is_file() else None
    if roles is None:
        pytest.skip("data/roles.toml is personal and not in the repository")
    expect = {"Analista Funcional Senior": "bsa-insurance", "Líder Técnico Java": "tech-lead-java",
              "Ingeniero de Soporte de Aplicaciones": "app-support-java", "Analista de Datos": "data-bi-analyst",
              "Analista QA": "qa-analyst", "Desarrollador Java": "java-backend"}
    for title, role_id in expect.items():
        assert match_role(title, "", roles).role.id == role_id


# ---- GetOnBoard ----------------------------------------------------------------------------------


def job(title: str = "Desarrollador Java", **attrs: Any) -> dict[str, Any]:
    base = {"title": title, "remote": True, "remote_modality": "fully_remote", "countries": ["Chile"],
            "description": "<p>Trabajo 100% remoto</p>", "published_at": 1794442211,
            "company": {"data": {"attributes": {"name": "Acme"}}}, "min_salary": 3000, "max_salary": 4000}
    return {"id": title, "attributes": {**base, **attrs}, "links": {"public_url": f"https://getonbrd.com/jobs/{title}"}}


def test_parse_job_keeps_only_remote_postings_open_to_chile() -> None:
    assert getonboard.parse_job(job()) is not None
    assert getonboard.parse_job(job(remote=False)) is None
    assert getonboard.parse_job(job(remote_modality="hybrid_remote")) is None
    assert getonboard.parse_job(job(countries=["United States"], remote_zone=None)) is None
    assert getonboard.parse_job(job(countries=["Colombia"], remote_zone="latam")) is not None
    parsed = getonboard.parse_job(job())
    assert parsed is not None and parsed.posting.company == "Acme" and "USD 3000 - 4000" in parsed.posting.text


def test_search_filters_and_adds_chile_rows(tmp_path: Path) -> None:
    path = create_tracker_template(tmp_path / "tracker.xlsx")
    postular.ensure_sheet(path)
    postular.table(path).append({"link": "https://getonbrd.com/jobs/Analista Funcional"})  # already tracked
    pages = {"data": [job("Desarrollador Java"), job("Analista Funcional"), job("Diseñador UX"),
                      job("Analista de Datos", description="<p>Modalidad híbrida, 2 días presenciales</p>"),
                      job("QA Senior", remote=False)], "meta": {"page": 1, "total_pages": 1}}
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=pages)))
    roles = load_roles(Path(__file__).resolve().parent.parent / "data") if (
        Path(__file__).resolve().parent.parent / "data" / "roles.toml").is_file() else None
    if roles is None:
        pytest.skip("data/roles.toml is personal and not in the repository")
    result = getonboard.search(path, roles, client, queries=("java",), delay=0)
    assert [c.posting.title for c in result.candidates] == ["Desarrollador Java"]
    assert result.discarded["ya está en el tracker"] == 1 and result.discarded["no es de tus familias de rol"] == 1
    assert result.discarded["texto menciona híbrido/presencial"] == 1
    [row] = getonboard.add_to_tracker(path, result.candidates)
    saved = next(r for r in postular.table(path).rows() if r["_row"] == row)
    assert saved["country"] == "Chile" and saved["status"] == "Nuevo" and saved["source"] == "getonboard"
    assert saved["contract"] == NOT_STATED and saved["salary_clp"] == NOT_STATED
    assert json.dumps(saved["description"]).count("Trabajo 100% remoto") == 1


# ---- verify ---------------------------------------------------------------------------------------


def _verify_es(master: CVMaster, header: str) -> list[str]:
    from job_assistant.cv.verify import verify_resume

    pdf_text = (f"ALEX SAMPLE\nDesarrollador Java Senior\n{header}\nRESUMEN PROFESIONAL\nTexto\n"
                "Stack técnico: Java")
    verdict = verify_resume("x", spanish_content(), "es", SPANISH_JD, 2, pdf_text, master)
    return [i.problem for i in verdict.issues if i.severity == "blocker"]


def test_verify_accepts_a_clean_chile_resume(master: CVMaster) -> None:
    assert _verify_es(master, "alex@example.com | linkedin.com/in/alex-sample | (000) 000-0000 | Chile") == []


def test_verify_blocks_address_or_authorization_in_a_chile_resume(master: CVMaster) -> None:
    problems = _verify_es(master, "alex@example.com | Toronto, ON, Canada\nCanadian Permanent Resident")
    assert any("mailing address" in p for p in problems) and any("work authorization" in p for p in problems)


def test_chile_resume_file_is_detected_as_spanish_by_name(tmp_path: Path) -> None:
    assert postular.cv_filename("A", "es", "Acme", "Analista") == "A_CV_ES_Acme_Analista.pdf"
    assert postular.cv_filename("A", "fr", "Acme", "Analyste") == "A_CV_Acme_Analyste.pdf"  # French unchanged
    assert postular.cv_filename("A", "en", "Acme", "Analyst") == "A_Resume_Acme_Analyst.pdf"  # English unchanged
