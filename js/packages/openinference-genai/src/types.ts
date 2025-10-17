export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

export type DeeplyMutable<T> = {
  -readonly [P in keyof T]: T[P] extends object ? DeeplyMutable<T[P]> : T[P];
};
