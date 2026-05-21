use super::*;

pub(super) fn validate_feff_consistency(input: &FeffInput, active_cards: &[String]) -> Result<()> {
    let spectroscopy_cards = [
        "XANES", "EXAFS", "XES", "DANES", "FPRIME", "ELNES", "EXELFS",
    ];
    let active_spectroscopy = spectroscopy_cards
        .iter()
        .filter(|card| active_card(active_cards, card))
        .count();
    if active_spectroscopy > 1
        && let Some(card) = spectroscopy_cards
            .iter()
            .find(|card| active_card(active_cards, card))
    {
        return Err(parse_error(
            required_card_line(input, card)?,
            "ERROR more than one type of spectroscopy selected",
        ));
    }

    if active_card(active_cards, "NRIXS") {
        let nrixs_line = required_card_line(input, "NRIXS")?;
        let xanes_or_exafs = ["XANES", "EXAFS"]
            .iter()
            .filter(|card| active_card(active_cards, card))
            .count();
        if xanes_or_exafs != 1 {
            return Err(parse_error(
                nrixs_line,
                "NRIXS must be combined with XANES or EXAFS",
            ));
        }
        if let Some(card) = ["FPRIME", "XES", "DANES", "ELNES", "EXELFS"]
            .iter()
            .find(|card| active_card(active_cards, card))
        {
            return Err(parse_error(
                required_card_line(input, card)?,
                "NRIXS combined with incompatible spectroscopy card",
            ));
        }
        if active_card(active_cards, "MULT") {
            return Err(parse_error(
                required_card_line(input, "MULT")?,
                "you cannot combine NRIXS and MULTIPOLE",
            ));
        }
        if let Some(card) = [
            "ELLIPTICITY",
            "POLARIZATION",
            "NSTAR",
            "SPIN",
            "CFAVERAGE",
            "XMCD",
            "RPHASES",
            "TDLDA",
            "PMBSE",
            "HUBBARD",
        ]
        .iter()
        .find(|card| active_card(active_cards, card))
        {
            return Err(parse_error(
                required_card_line(input, card)?,
                "card is explicitly forbidden for NRIXS",
            ));
        }
    } else if active_card(active_cards, "LJMAX") || active_card(active_cards, "LDECMX") {
        let line = card_by_feff_name(input, "LJMAX")
            .or_else(|| card_by_feff_name(input, "LDECMX"))
            .ok_or_else(|| IoError::Parse {
                path: input.source.clone(),
                line: 0,
                message: "LDEC/LJMAX card not found".to_string(),
            })?;
        return Err(parse_error(
            line,
            "LDEC and LJMAX cards only allowed with NRIXS",
        ));
    }

    if active_card(active_cards, "RECIPROCAL") {
        let reciprocal_line = required_card_line(input, "RECIPROCAL")?;
        if !(active_card(active_cards, "KMESH") && active_card(active_cards, "TARGET")) {
            return Err(parse_error(
                reciprocal_line,
                "KMESH and TARGET are required for RECIPROCAL card",
            ));
        }

        let structure_source_count = ["LATTICE", "CIF"]
            .iter()
            .filter(|card| active_card(active_cards, card))
            .count();
        if structure_source_count != 1 {
            return Err(parse_error(
                reciprocal_line,
                "use either LATTICE or CIF with RECIPROCAL card",
            ));
        }
    }

    if active_card(active_cards, "CGRID")
        && !(active_card(active_cards, "COMPTON") || active_card(active_cards, "RHOZZP"))
    {
        return Err(parse_error(
            required_card_line(input, "CGRID")?,
            "Cannot use CGRID without COMPTON or RHOZZP.  Exiting.",
        ));
    }

    if active_card(active_cards, "HUBBARD") && active_card(active_cards, "RECIPROCAL") {
        return Err(parse_error(
            required_card_line(input, "HUBBARD")?,
            "Cannot use RECIPROCAL with HUBBARD.",
        ));
    }

    Ok(())
}
