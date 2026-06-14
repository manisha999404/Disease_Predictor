import { QueryClient } from "@tanstack/react-query";
import { createRouter } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";

export const getRouter = () => {
  const queryClient = new QueryClient();

  const router = createRouter({
    routeTree,
    context: { queryClient },
    // scrollRestoration disabled — it attaches a focus listener to the entire
    // document that synchronously reads getBoundingClientRect() + writes to
    // sessionStorage on every input focus, causing a main thread spike = freeze
    scrollRestoration: false,
    defaultPreloadStaleTime: 0,
  });

  return router;
};