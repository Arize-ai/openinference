const baseLangchainMessage = {
  lc_id: ["human"],
  lc_kwargs: {
    content: "hello, this is a test",
  },
};
export const getLangchainMessage = (
  config?: Partial<{
    lc_id: string[];
    lc_kwargs: Record<string, unknown>;
  }>,
) => {
  return Object.assign({ ...baseLangchainMessage }, config);
};
