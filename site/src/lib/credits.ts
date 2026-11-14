/**
 * Third-party work used on the site, listed on /credits.html. Each entry
 * carries what its licence asks for (title, author, source, licence and any
 * changes), and public-domain work is credited all the same.
 */
export interface Credit {
  /** What it is used for on the site. */
  use: string;
  title: string;
  url: string;
  author: string;
  authorUrl?: string;
  license: string;
  licenseUrl?: string;
  /** How it was changed, if it was. */
  changes?: string;
}

export const photos: Credit[] = [
  {
    use: "Home page hero",
    title: "Cupola Windows (iss038e013587)",
    url: "https://images.nasa.gov/details/iss038e013587",
    author: "NASA",
    license: "Public domain",
    changes:
      "The Cupola aboard the International Space Station, photographed on 8 December 2013. Cropped and printed as a halftone screen, with the OpenInference lockup drawn over it.",
  },
];

/** The hero photo, for its caption. */
export const heroPhoto = photos[0];

export const typefaces: Credit[] = [
  {
    use: "Headings and body text",
    title: "Urbanist",
    url: "https://github.com/coreyhu/Urbanist",
    author: "Corey Hu",
    license: "SIL Open Font License 1.1",
    licenseUrl: "https://openfontlicense.org/",
  },
  {
    use: "Labels and code",
    title: "Geist Mono",
    url: "https://github.com/vercel/geist-font",
    author: "Vercel",
    license: "SIL Open Font License 1.1",
    licenseUrl: "https://openfontlicense.org/",
  },
];
