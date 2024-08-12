/**
 * Icons sourced from @see https://feathericons.com/
 */

export function ThumbsUp({ filled }: { filled?: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill={filled ? "#666b71" : "none"}
      stroke="#666b71"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      className="feather feather-thumbs-up"
    >
      <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
    </svg>
  );
}

export function ThumbsDown({ filled }: { filled?: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill={filled ? "#666b71" : "none"}
      stroke="#666b71"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      className="feather feather-thumbs-down gray-600"
    >
      <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>
    </svg>
  );
}
