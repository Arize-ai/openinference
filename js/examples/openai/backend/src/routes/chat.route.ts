import express, { Router } from "express";
import { chat } from "../controllers/chat.controller";

const router: Router = express.Router();

router.route("/").post(chat);

export default router;
