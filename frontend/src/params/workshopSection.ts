/**
 * The route parameter matcher of the Workshop settings pages
 * (/workshop/[section=workshopSection]): a segment matches when the
 * destination table has a Workshop settings page by that name.
 */

import type { ParamMatcher } from '@sveltejs/kit';
import { isWorkshopSection } from '$lib/nav/destinations';

export const match: ParamMatcher = (param) => isWorkshopSection(param);
