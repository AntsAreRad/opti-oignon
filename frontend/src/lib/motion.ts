/**
 * The scroll behaviour the motion preference allows.
 *
 * The theme path (lib/theme/apply.ts) marks the root element with one of
 * two classes, from the motion preference: a choice to reduce motion, or a
 * choice of full motion whatever the system says. With neither, the system's reduced-motion
 * setting decides. Scrolling is smooth only when nothing reduces motion,
 * and it is not smooth when the environment cannot be read (no document).
 *
 * Every scroll that sets a behaviour takes it from scrollBehavior(), so the
 * choice is made here and nowhere else.
 */

/** The class the motion preference sets when the user chose reduced motion. */
export const MOTION_REDUCED_CLASS = 'oo-reduce-motion';

/** The class the motion preference sets when the user chose full motion. */
export const MOTION_FULL_CLASS = 'oo-motion-full';

export interface MotionEnvironment {
	/** The user chose reduced motion. */
	reducedByChoice: boolean;
	/** The user chose full motion, whatever the system says. */
	fullByChoice: boolean;
	/** The system asks for reduced motion. */
	reducedBySystem: boolean;
}

/** The motion environment of the page, or null where there is no page. */
export function readMotionEnvironment(): MotionEnvironment | null {
	if (typeof document === 'undefined' || typeof window === 'undefined') return null;
	const root = document.documentElement;
	const query =
		typeof window.matchMedia === 'function'
			? window.matchMedia('(prefers-reduced-motion: reduce)')
			: null;
	return {
		reducedByChoice: root.classList.contains(MOTION_REDUCED_CLASS),
		fullByChoice: root.classList.contains(MOTION_FULL_CLASS),
		reducedBySystem: query ? query.matches : true,
	};
}

/** 'smooth' when motion is allowed, 'auto' when it is reduced or unknown. */
export function scrollBehavior(
	environment: MotionEnvironment | null = readMotionEnvironment()
): 'auto' | 'smooth' {
	if (!environment) return 'auto';
	if (environment.reducedByChoice) return 'auto';
	if (environment.fullByChoice) return 'smooth';
	return environment.reducedBySystem ? 'auto' : 'smooth';
}
